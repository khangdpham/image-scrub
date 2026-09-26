#!/usr/bin/env python3
"""
image_scrub.py - Strip privacy metadata from images while keeping pixels identical.

Supported (detected by file content, not extension):
  JPEG   .jpg .jpeg .jpe .jfif   lossless: segments removed, image data copied byte-for-byte
  PNG    .png (incl. APNG)       lossless: metadata chunks removed
  WebP   .webp (incl. animated)  lossless: EXIF/XMP chunks removed, header fixed up
  GIF    .gif (incl. animated)   lossless: comment/XMP application blocks removed
  HEIC   .heic .heif .avif       lossless: Exif/XMP/C2PA items blanked in place
  TIFF   .tif .tiff              rebuilt from decoded pixels with a lossless codec
                                 (needs Pillow: pip3 install pillow)
  BMP    .bmp                    copied (BMP carries no identifying metadata)

Removed: EXIF (GPS, camera make/model/serial, dates, thumbnails), XMP, IPTC,
comments, text chunks, C2PA / Content Credentials manifests, MPF / depth maps /
gain maps, data appended after the image, and macOS extended attributes
(quarantine, "Where From" URL) because a brand-new file is written.

Kept by default (no personal info, but they affect how the image looks):
  - ICC color profile        (--strip-icc to remove; colors may shift)
  - Orientation, as a single tag (--no-orientation to drop; photos may turn sideways)
  - Structural/rendering data: palettes, transparency, gamma, DPI, animation timing,
    HEIF rotation/mirror properties

Usage:
  python3 image_scrub.py photo.heic                 -> photo_clean.heic
  python3 image_scrub.py a.jpg b.png -o clean/      -> clean/a.jpg, clean/b.png
  python3 image_scrub.py ~/Pictures/trip -o clean/  (whole folder, recursive)
  python3 image_scrub.py photo.jpg --verify         pixel check (needs Pillow;
                                                    HEIC/AVIF also need pillow-heif)
"""

import argparse
import os
import struct
import sys
import zlib

SUPPORTED_EXT = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".apng", ".webp", ".gif",
                 ".heic", ".heif", ".hif", ".avif", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------- shared helpers

def read_orientation_tiff(t: bytes):
    """Orientation (1-8) from raw TIFF/EXIF bytes (starting at II/MM), or None."""
    if len(t) < 8 or t[:2] not in (b"II", b"MM"):
        return None
    e = "<" if t[:2] == b"II" else ">"
    try:
        ifd = struct.unpack(e + "I", t[4:8])[0]
        n = struct.unpack(e + "H", t[ifd:ifd + 2])[0]
        for i in range(n):
            off = ifd + 2 + i * 12
            tag, typ, cnt = struct.unpack(e + "HHI", t[off:off + 8])
            if tag == 0x0112 and typ == 3 and cnt == 1:
                v = struct.unpack(e + "H", t[off + 8:off + 10])[0]
                return v if 1 <= v <= 8 else None
    except struct.error:
        pass
    return None


def minimal_orientation_tiff(orientation: int) -> bytes:
    """A TIFF blob holding only the Orientation tag."""
    return (b"MM\x00\x2a" + struct.pack(">I", 8) + struct.pack(">H", 1)
            + struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0)
            + struct.pack(">I", 0))


def detect(data: bytes):
    if data[:3] == b"\xFF\xD8\xFF":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[4:8] == b"ftyp":
        return "isobmff"
    if data[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff"
    if data[:2] == b"BM":
        return "bmp"
    return None


# ---------------------------------------------------------------- JPEG

J_EOI, J_SOS = 0xD9, 0xDA
J_RST = range(0xD0, 0xD8)
J_STRUCTURAL = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD,
                0xCE, 0xCF, 0xC4, 0xCC, 0xDB, 0xDD, 0xDC, 0xDE, 0xDF, J_SOS}


def _jpeg_keep(marker, payload, keep_icc):
    if marker in J_STRUCTURAL:
        return True
    if marker == 0xE0:
        return payload.startswith(b"JFIF\x00")
    if marker == 0xE2:
        return keep_icc and payload.startswith(b"ICC_PROFILE\x00")
    if marker == 0xEE:
        return payload.startswith(b"Adobe")       # needed to decode CMYK/YCCK
    return False


def _jpeg_describe(marker, payload):
    if marker == 0xE1 and payload.startswith(b"Exif"):
        return "EXIF (camera, dates, GPS, thumbnail)"
    if marker == 0xE1 and b"ns.adobe.com/xap" in payload[:64]:
        return "XMP metadata"
    if marker == 0xE1 and b"ns.adobe.com/xmp/extension" in payload[:64]:
        return "Extended XMP"
    if marker == 0xE2 and payload.startswith(b"MPF"):
        return "MPF (multi-picture index)"
    if marker == 0xE2 and payload.startswith(b"ICC_PROFILE"):
        return "ICC profile"
    if marker == 0xEB and (b"jumb" in payload[:64] or b"c2pa" in payload[:128]):
        return "C2PA / Content Credentials manifest"
    if marker == 0xED:
        return "IPTC / Photoshop (APP13)"
    if marker == 0xFE:
        return "comment"
    if 0xE0 <= marker <= 0xEF:
        tag = payload[:12].split(b"\x00")[0].decode("latin-1", "replace")
        return f"APP{marker - 0xE0} segment ({tag or 'unnamed'})"
    return f"marker 0x{marker:02X}"


def scrub_jpeg(data, keep_icc, keep_orientation):
    out = bytearray(b"\xFF\xD8")
    removed, orientation, inserted = [], None, False
    i, n, in_scan = 2, len(data), False
    while i < n:
        if in_scan:
            j = i
            while j < n - 1:
                if data[j] == 0xFF:
                    nxt = data[j + 1]
                    if nxt == 0x00 or nxt in J_RST:
                        j += 2
                        continue
                    if nxt == 0xFF:
                        j += 1
                        continue
                    break
                j += 1
            else:
                j = n
            out += data[i:j]
            i, in_scan = j, False
            continue
        if data[i] != 0xFF:
            raise ValueError(f"corrupt JPEG: expected marker at offset {i}")
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = data[i]
        i += 1
        if marker == J_EOI:
            out += b"\xFF\xD9"
            if i < n:
                removed.append(f"{n - i:,} bytes after image (MPF images/gain map/etc.)")
            return bytes(out), removed, orientation
        if marker in J_RST or marker == 0x01:
            out += bytes([0xFF, marker])
            continue
        length = struct.unpack(">H", data[i:i + 2])[0]
        end = i + length
        payload = data[i + 2:end]
        if marker == 0xE1 and orientation is None and payload.startswith(b"Exif\x00\x00"):
            orientation = read_orientation_tiff(payload[6:])
        if marker in J_STRUCTURAL and not inserted:
            if keep_orientation and orientation and orientation != 1:
                p = b"Exif\x00\x00" + minimal_orientation_tiff(orientation)
                out += b"\xFF\xE1" + struct.pack(">H", len(p) + 2) + p
            inserted = True
        if _jpeg_keep(marker, payload, keep_icc):
            out += b"\xFF" + bytes([marker]) + data[i:end]
        else:
            removed.append(_jpeg_describe(marker, payload))
        i = end
        if marker == J_SOS:
            in_scan = True
    out += b"\xFF\xD9"
    removed.append("file was truncated (no end marker); end marker appended")
    return bytes(out), removed, orientation


# ---------------------------------------------------------------- PNG

PNG_KEEP = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"gAMA", b"cHRM", b"sRGB",
            b"sBIT", b"pHYs", b"bKGD", b"hIST", b"sPLT", b"cICP", b"mDCv", b"cLLi",
            b"acTL", b"fcTL", b"fdAT"}
PNG_NAMES = {b"tEXt": "text chunk", b"zTXt": "compressed text chunk",
             b"iTXt": "international text chunk (often XMP)", b"eXIf": "EXIF",
             b"tIME": "modification time", b"iCCP": "ICC profile",
             b"caBX": "C2PA / Content Credentials manifest"}


def _png_chunk(typ, payload):
    return (struct.pack(">I", len(payload)) + typ + payload
            + struct.pack(">I", zlib.crc32(typ + payload) & 0xFFFFFFFF))


def scrub_png(data, keep_icc, keep_orientation):
    out = bytearray(data[:8])
    removed, orientation, inserted = [], None, False
    i, n = 8, len(data)
    # pre-scan for orientation in eXIf
    j = 8
    while j + 8 <= n:
        ln = struct.unpack(">I", data[j:j + 4])[0]
        if data[j + 4:j + 8] == b"eXIf":
            orientation = read_orientation_tiff(data[j + 8:j + 8 + ln])
        j += 12 + ln
    while i + 8 <= n:
        ln = struct.unpack(">I", data[i:i + 4])[0]
        typ = data[i + 4:i + 8]
        end = i + 12 + ln
        if typ == b"IDAT" and not inserted:
            if keep_orientation and orientation and orientation != 1:
                out += _png_chunk(b"eXIf", minimal_orientation_tiff(orientation))
            inserted = True
        keep = typ in PNG_KEEP or (typ == b"iCCP" and keep_icc)
        if not keep and typ not in PNG_NAMES and typ[0:1].isupper():
            keep = True   # unknown *critical* chunk: dropping it could break decoding
        if keep:
            out += data[i:end]
        else:
            removed.append(PNG_NAMES.get(typ, f"chunk {typ.decode('latin-1')}"))
        i = end
        if typ == b"IEND":
            if i < n:
                removed.append(f"{n - i:,} bytes after image")
            break
    return bytes(out), removed, orientation


# ---------------------------------------------------------------- WebP

WEBP_KEEP = {b"VP8 ", b"VP8L", b"VP8X", b"ALPH", b"ANIM", b"ANMF"}
WEBP_NAMES = {b"EXIF": "EXIF", b"XMP ": "XMP metadata", b"ICCP": "ICC profile",
              b"C2PA": "C2PA / Content Credentials manifest"}


def scrub_webp(data, keep_icc, keep_orientation):
    riff_end = min(len(data), 8 + struct.unpack("<I", data[4:8])[0])
    chunks, removed, has_icc = [], [], False
    i = 12
    while i + 8 <= riff_end:
        typ = data[i:i + 4]
        ln = struct.unpack("<I", data[i + 4:i + 8])[0]
        end = i + 8 + ln + (ln & 1)
        if typ in WEBP_KEEP or (typ == b"ICCP" and keep_icc):
            chunks.append(bytearray(data[i:end]))
            has_icc |= typ == b"ICCP"
        else:
            removed.append(WEBP_NAMES.get(typ, f"chunk {typ.decode('latin-1').strip()}"))
        i = end
    for c in chunks:                       # fix VP8X feature flags
        if c[:4] == b"VP8X":
            c[8] &= ~0x0C & 0xFF           # clear EXIF (0x08) and XMP (0x04)
            if not has_icc:
                c[8] &= ~0x20 & 0xFF       # clear ICC
    body = b"WEBP" + b"".join(chunks)
    if len(data) > riff_end:
        removed.append(f"{len(data) - riff_end:,} bytes after image")
    # WebP rarely carries orientation and decoders ignore it, so nothing to keep
    return b"RIFF" + struct.pack("<I", len(body)) + body, removed, None


# ---------------------------------------------------------------- GIF

GIF_KEEP_APPS = {b"NETSCAPE2.0", b"ANIMEXTS1.0"}   # looping info


def _gif_subblocks_end(data, i):
    while True:
        sz = data[i]
        i += 1
        if sz == 0:
            return i
        i += sz


def scrub_gif(data, keep_icc, keep_orientation):
    out = bytearray(data[:13])
    removed = []
    flags = data[10]
    i = 13
    if flags & 0x80:
        tbl = 3 * (2 ** ((flags & 7) + 1))
        out += data[i:i + tbl]
        i += tbl
    n = len(data)
    while i < n:
        b = data[i]
        if b == 0x3B:                                   # trailer
            out += b"\x3B"
            i += 1
            if i < n:
                removed.append(f"{n - i:,} bytes after image")
            break
        if b == 0x2C:                                   # image descriptor
            start = i
            lflags = data[i + 9]
            i += 10
            if lflags & 0x80:
                i += 3 * (2 ** ((lflags & 7) + 1))
            i = _gif_subblocks_end(data, i + 1)         # +1 = LZW min code size
            out += data[start:i]
        elif b == 0x21:                                 # extension
            label, start = data[i + 1], i
            end = _gif_subblocks_end(data, i + 2)
            if label in (0xF9, 0x01):                   # graphic control, plain text
                out += data[start:end]
            elif label == 0xFF:
                ident = data[i + 3:i + 3 + data[i + 2]]
                if ident in GIF_KEEP_APPS or (keep_icc and ident == b"ICCRGBG1012"):
                    out += data[start:end]
                elif ident.startswith(b"XMP Data"):
                    removed.append("XMP metadata")
                else:
                    removed.append(f"application block ({ident.decode('latin-1', 'replace')})")
            elif label == 0xFE:
                removed.append("comment")
            else:
                removed.append(f"extension 0x{label:02X}")
            i = end
        else:
            raise ValueError(f"corrupt GIF at offset {i}")
    else:
        out += b"\x3B"
        removed.append("file was truncated; trailer appended")
    return bytes(out), removed, None


# ---------------------------------------------------------------- HEIC / HEIF / AVIF

def _boxes(data, start, end):
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        typ = bytes(data[pos + 4:pos + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end:
            break
        yield typ, pos, hdr, pos + size
        pos += size


def _read_uint(data, pos, size):
    if size == 0:
        return 0, pos
    return int.from_bytes(data[pos:pos + size], "big"), pos + size


def _parse_iinf(data, s, e):
    items = {}
    ver = data[s]
    p = s + 4
    cnt = struct.unpack(">H" if ver == 0 else ">I", data[p:p + (2 if ver == 0 else 4)])[0]
    p += 2 if ver == 0 else 4
    for typ, bs, hdr, be in _boxes(data, p, e):
        if typ != b"infe":
            continue
        q = bs + hdr
        v = data[q]
        q += 4
        if v < 2:
            continue
        if v == 2:
            iid = struct.unpack(">H", data[q:q + 2])[0]; q += 2
        else:
            iid = struct.unpack(">I", data[q:q + 4])[0]; q += 4
        q += 2                                              # protection index
        itype = bytes(data[q:q + 4]); q += 4
        q = data.index(b"\x00", q, be) + 1                  # item_name
        ctype = b""
        if itype == b"mime" and q < be:
            z = data.find(b"\x00", q, be)
            ctype = bytes(data[q:z if z != -1 else be])
        items[iid] = (itype, ctype)
    return items


def _parse_iloc(data, s, e):
    locs = {}
    ver = data[s]
    p = s + 4
    off_sz, len_sz = data[p] >> 4, data[p] & 15
    base_sz, idx_sz = data[p + 1] >> 4, (data[p + 1] & 15) if ver in (1, 2) else 0
    p += 2
    if ver < 2:
        cnt = struct.unpack(">H", data[p:p + 2])[0]; p += 2
    else:
        cnt = struct.unpack(">I", data[p:p + 4])[0]; p += 4
    for _ in range(cnt):
        if ver < 2:
            iid = struct.unpack(">H", data[p:p + 2])[0]; p += 2
        else:
            iid = struct.unpack(">I", data[p:p + 4])[0]; p += 4
        method = 0
        if ver in (1, 2):
            method = struct.unpack(">H", data[p:p + 2])[0] & 15; p += 2
        p += 2                                              # data_reference_index
        base, p = _read_uint(data, p, base_sz)
        ecnt = struct.unpack(">H", data[p:p + 2])[0]; p += 2
        exts = []
        for _ in range(ecnt):
            _, p = _read_uint(data, p, idx_sz)
            off, p = _read_uint(data, p, off_sz)
            ln, p = _read_uint(data, p, len_sz)
            exts.append((base + off, ln))
        locs[iid] = (method, exts)
    return locs


EMPTY_HEIF_EXIF = struct.pack(">I", 0) + b"MM\x00\x2a\x00\x00\x00\x08" + b"\x00" * 6
EMPTY_XMP = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"/>'
C2PA_UUID = bytes.fromhex("d8fec3d61b0e483c92975828877ec481")
XMP_UUID = bytes.fromhex("be7acfcb97a942e89c71999491e3afac")


def scrub_isobmff(data, keep_icc, keep_orientation):
    buf = bytearray(data)
    removed = []
    meta = None
    for typ, bs, hdr, be in _boxes(buf, 0, len(buf)):
        if typ == b"meta":
            meta = (bs + hdr, be)
        elif typ == b"uuid":
            u = bytes(buf[bs + hdr:bs + hdr + 16])
            what = {C2PA_UUID: "C2PA / Content Credentials manifest",
                    XMP_UUID: "XMP metadata"}.get(u, "vendor uuid box")
            buf[bs + 4:bs + 8] = b"free"                  # same size, readers skip it
            buf[bs + hdr:be] = bytes(be - bs - hdr)
            removed.append(what)
        elif typ == b"moov":
            raise ValueError("this is a video/sequence file, not a still image")
    if not meta:
        raise ValueError("no 'meta' box; not a HEIF/AVIF still image")
    ms, me = meta
    iinf = iloc = idat = None
    for typ, bs, hdr, be in _boxes(buf, ms + 4, me):      # meta is a FullBox
        if typ == b"iinf":
            iinf = (bs + hdr, be)
        elif typ == b"iloc":
            iloc = (bs + hdr, be)
        elif typ == b"idat":
            idat = bs + hdr
    if not (iinf and iloc):
        raise ValueError("HEIF item tables missing")
    items = _parse_iinf(buf, *iinf)
    locs = _parse_iloc(buf, *iloc)
    for iid, (itype, ctype) in items.items():
        c = ctype.lower()
        if itype == b"Exif":
            what, filler = "EXIF (camera, dates, GPS)", EMPTY_HEIF_EXIF
        elif itype == b"mime" and b"xml" in c:
            what, filler = "XMP metadata", EMPTY_XMP
        elif itype == b"c2pa" or (itype == b"mime" and b"c2pa" in c):
            what, filler = "C2PA / Content Credentials manifest", b""
        else:
            continue
        if iid not in locs:
            continue
        method, exts = locs[iid]
        if method not in (0, 1) or (method == 1 and idat is None):
            removed.append(f"{what}: unsupported storage, NOT removed")
            continue
        first = True
        for off, ln in exts:
            a = off + (idat if method == 1 else 0)
            if ln == 0 or a + ln > len(buf):
                continue
            fill = filler if first and len(filler) <= ln else b""
            buf[a:a + ln] = fill + (b" " if filler is EMPTY_XMP else b"\x00") * (ln - len(fill))
            first = False
        removed.append(what)
    # HEIF orientation lives in irot/imir properties (kept); ICC lives in colr (kept)
    return bytes(buf), removed, None


# ---------------------------------------------------------------- TIFF (via Pillow)

def scrub_tiff(src, dst, keep_icc, keep_orientation):
    try:
        from PIL import Image, ImageSequence
    except ImportError:
        raise ValueError("TIFF needs Pillow: pip3 install pillow")
    im = Image.open(src)
    T = Image.Transpose
    undo = {2: T.FLIP_LEFT_RIGHT, 3: T.ROTATE_180, 4: T.FLIP_TOP_BOTTOM,
            5: T.TRANSPOSE, 6: T.ROTATE_90, 7: T.TRANSVERSE, 8: T.ROTATE_270}
    frames, orientation = [], None
    for f in ImageSequence.Iterator(im):
        o = f.getexif().get(0x0112)                 # read BEFORE load()
        if orientation is None:
            orientation = o
        f.load()
        clean = Image.frombytes(f.mode, f.size, f.tobytes())
        if f.mode in ("P", "PA"):
            clean.putpalette(f.getpalette())
        # Pillow auto-rotates TIFFs on load (and drops the tag). If we keep the
        # tag, undo that rotation so stored pixels match the original exactly;
        # if we drop the tag, keep the rotated pixels so it still displays upright.
        if keep_orientation and o in undo and f.getexif().get(0x0112) is None:
            clean = clean.transpose(undo[o])
        frames.append(clean)
    comp = im.info.get("compression", "raw")
    if comp not in ("raw", "tiff_lzw", "tiff_adobe_deflate", "packbits"):
        comp = "tiff_lzw"   # re-encode lossy-compressed TIFFs losslessly
    kw = {"compression": comp}
    if keep_icc and im.info.get("icc_profile"):
        kw["icc_profile"] = im.info["icc_profile"]
    if keep_orientation and orientation and orientation != 1:
        kw["tiffinfo"] = {0x0112: orientation}
    frames[0].save(dst, format="TIFF", save_all=len(frames) > 1,
                   append_images=frames[1:], **kw)
    return ["all TIFF tags except image structure"
            + (", ICC" if keep_icc else "") + (", orientation" if keep_orientation else "")
            + " (EXIF, GPS, XMP, IPTC, artist, software, dates...)"], orientation


# ---------------------------------------------------------------- verify

def verify(src, dst):
    try:
        from PIL import Image, ImageSequence
    except ImportError:
        print("  verify: Pillow not installed (pip3 install pillow); skipped")
        return
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        if hasattr(pillow_heif, "register_avif_opener"):
            pillow_heif.register_avif_opener()
    except ImportError:
        pass
    try:
        a, b = Image.open(src), Image.open(dst)
        fa = [f.copy() for f in ImageSequence.Iterator(a)]
        fb = [f.copy() for f in ImageSequence.Iterator(b)]
    except Exception as e:
        print(f"  verify: could not decode ({e}); for HEIC/AVIF: pip3 install pillow-heif")
        return
    same = len(fa) == len(fb) and all(
        x.size == y.size and x.mode == y.mode and x.tobytes() == y.tobytes()
        for x, y in zip(fa, fb))
    if same:
        extra = f" across {len(fa)} frames" if len(fa) > 1 else ""
        print(f"  verify: PASS - decoded pixels are identical{extra}")
    else:
        print("  verify: FAIL - pixels differ!")
        sys.exit(2)


# ---------------------------------------------------------------- main

SCRUBBERS = {"jpeg": scrub_jpeg, "png": scrub_png, "webp": scrub_webp,
             "gif": scrub_gif, "isobmff": scrub_isobmff}


def process(src, dst, args):
    if os.path.abspath(dst) == os.path.abspath(src):
        print(f"{src}: refusing to overwrite the original; pick another output")
        return
    with open(src, "rb") as f:
        data = f.read()
    kind = detect(data)
    keep_icc, keep_or = not args.strip_icc, not args.no_orientation
    try:
        if kind in SCRUBBERS:
            clean, removed, orient = SCRUBBERS[kind](data, keep_icc, keep_or)
            with open(dst, "wb") as f:          # fresh file: no xattrs / Finder info
                f.write(clean)
        elif kind == "tiff":
            removed, orient = scrub_tiff(src, dst, keep_icc, keep_or)
        elif kind == "bmp":
            with open(dst, "wb") as f:
                f.write(data)
            removed, orient = [], None
        else:
            print(f"{src}: skipped (unrecognized image format)")
            return
    except (ValueError, IndexError, struct.error) as e:
        print(f"{src}: ERROR - {e}")
        return
    print(f"{src} -> {dst}  [{kind}]  ({len(data):,} -> {os.path.getsize(dst):,} bytes)")
    for r in removed or ["(nothing to remove)"]:
        print(f"  removed: {r}")
    if orient and orient != 1 and keep_or and kind != "tiff":
        print(f"  kept: orientation={orient} (only tag retained)")
    if args.verify:
        verify(src, dst)


def expand(inputs):
    for p in inputs:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in sorted(files):
                    if os.path.splitext(name)[1].lower() in SUPPORTED_EXT:
                        yield os.path.join(root, name), p
        else:
            yield p, None


def main():
    ap = argparse.ArgumentParser(description="Strip privacy metadata from images, keeping pixels identical.")
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("-o", "--output", help="output file (single input) or folder")
    ap.add_argument("--strip-icc", action="store_true", help="also remove ICC color profile")
    ap.add_argument("--no-orientation", action="store_true", help="drop orientation too")
    ap.add_argument("--verify", action="store_true", help="confirm pixels match (needs Pillow)")
    args = ap.parse_args()

    jobs = list(expand(args.inputs))
    to_dir = args.output and (len(jobs) > 1 or os.path.isdir(args.output)
                              or args.output.endswith(os.sep))
    for src, base in jobs:
        if args.output and to_dir:
            rel = os.path.relpath(src, base) if base else os.path.basename(src)
            dst = os.path.join(args.output, rel)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        elif args.output:
            dst = args.output
        else:
            root, ext = os.path.splitext(src)
            dst = f"{root}_clean{ext}"
        process(src, dst, args)


if __name__ == "__main__":
    main()
