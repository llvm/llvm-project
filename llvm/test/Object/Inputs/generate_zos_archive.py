#!/usr/bin/env python3
"""Generate z/OS archive files

z/OS archives use EBCDIC encoding for headers, magic bytes, and symbol names.
This script generates archives in place to avoid reliance on canned binaries.

Usage examples:
  # Valid archive with one member and symbol table:
  %python %S/Inputs/generate_zos_archive.py --output %t.a \
      --symtab "foo:0" --member foo.o:%S/Inputs/foo.o

  # Empty archive:
  %python %S/Inputs/generate_zos_archive.py --output %t.a --empty

  # Malformed member header: bad terminator
  %python %S/Inputs/generate_zos_archive.py --output %t.a \
      --member foo.o --bad-terminator

  # Malformed __.SYMDEF header: bad terminator
  %python %S/Inputs/generate_zos_archive.py --output %t.a \
      --member foo.o --symtab foo:0 --malform-symtab-hdr bad-terminator

  # Member with explicit hex content:
  %python %S/Inputs/generate_zos_archive.py --output %t.a \
      --member foo.o:hex:deadbeef
"""

import argparse
import struct
import sys
import os

# EBCDIC / ASCII conversion table.
# fmt: off
ASCII_TO_EBCDIC_TABLE = (
    0x00,0x01,0x02,0x03,0x37,0x2D,0x2E,0x2F,0x16,0x05,0x15,0x0B,0x0C,0x0D,0x0E,0x0F,
    0x10,0x11,0x12,0x13,0x3C,0x3D,0x32,0x26,0x18,0x19,0x3F,0x27,0x1C,0x1D,0x1E,0x1F,
    0x40,0x5A,0x7F,0x7B,0x5B,0x6C,0x50,0x7D,0x4D,0x5D,0x5C,0x4E,0x6B,0x60,0x4B,0x61,
    0xF0,0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,0xF9,0x7A,0x5E,0x4C,0x7E,0x6E,0x6F,
    0x7C,0xC1,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xD1,0xD2,0xD3,0xD4,0xD5,0xD6,
    0xD7,0xD8,0xD9,0xE2,0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xAD,0xE0,0xBD,0x5F,0x6D,
    0x79,0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x91,0x92,0x93,0x94,0x95,0x96,
    0x97,0x98,0x99,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xC0,0x4F,0xD0,0xA1,0x07,
)
# fmt: on


def ascii_to_ebcdic(s):
    """Convert an ASCII string/bytes to EBCDIC (IBM-1047)."""
    if isinstance(s, str):
        s = s.encode("ascii")
    return bytes(ASCII_TO_EBCDIC_TABLE[b] for b in s)


def ebcdic_pad(s, width, pad_char=" "):
    """Convert ASCII string to EBCDIC, right-padded with EBCDIC spaces."""
    ascii_padded = s.ljust(width, pad_char)
    return ascii_to_ebcdic(ascii_padded)


# z/OS archive magic: "!<arch>\n" in EBCDIC.
ZOS_MAGIC = b"\x5a\x4c\x81\x99\x83\x88\x6e\x15"

# Terminator: "`\n" in EBCDIC.
ZOS_TERMINATOR = b"\x79\x15"

# EBCDIC newline for padding.
EBCDIC_NEWLINE = b"\x15"


def make_member_header(
    name,
    modtime,
    uid,
    gid,
    mode,
    size,
    bad_terminator=False,
    empty_name=False,
    empty_uid=False,
    empty_gid=False,
    empty_modtime=False,
    empty_mode=False,
):
    """Build a 60-byte z/OS archive member header.

    Fields (all EBCDIC, space-padded):
      ar_name:  16 bytes
      ar_date:  12 bytes
      ar_uid:    6 bytes
      ar_gid:    6 bytes
      ar_mode:   8 bytes
      ar_size:  10 bytes
      ar_fmag:   2 bytes (terminator)
    Total: 60 bytes
    """
    # Handle long names.
    long_name_ext = b""
    if len(name) > 16:
        name_ebcdic = ascii_to_ebcdic(name)
        ext_len = len(name_ebcdic)
        display_name = "#1/%d" % ext_len
        long_name_ext = name_ebcdic
        # The size field includes the extended name length.
        size = size + ext_len
    else:
        display_name = name

    if empty_name:
        hdr = ebcdic_pad(" ", 16)
    else:
        hdr = ebcdic_pad(display_name, 16)

    if empty_modtime:
        hdr += ebcdic_pad("", 12)
    else:
        hdr += ebcdic_pad(str(modtime), 12)

    if empty_uid:
        hdr += ebcdic_pad("", 6)
    else:
        hdr += ebcdic_pad(str(uid), 6)

    if empty_gid:
        hdr += ebcdic_pad("", 6)
    else:
        hdr += ebcdic_pad(str(gid), 6)

    if empty_mode:
        hdr += ebcdic_pad("", 8)
    else:
        hdr += ebcdic_pad(str(mode), 8)

    hdr += ebcdic_pad(str(size), 10)

    if bad_terminator:
        hdr += b"\x00\x00"
    else:
        hdr += ZOS_TERMINATOR

    assert len(hdr) == 60, f"Header is {len(hdr)} bytes, expected 60"
    return hdr + long_name_ext


def make_symtab(symbols, member_offsets, truncated=False, bad_count=False):
    """Build a __.SYMDEF symbol table body.

    symbols: list of (symbol_name_ascii, member_index, attributes)
    member_offsets: list of offsets for each member (indexed by member_index)

    Format:
      4 bytes: number of symbols (big-endian)
      For each symbol: 4 bytes offset + 4 bytes attributes (big-endian)
      Null-terminated symbol names in EBCDIC
    """
    num_syms = len(symbols)
    if bad_count:
        # Write a count that exceeds the buffer.
        body = struct.pack(">I", 0xFFFFFFFF)
    else:
        body = struct.pack(">I", num_syms)

    if truncated:
        # Return just the count, truncated before offset table.
        return body[:2]

    for sym_name, mem_idx, attrs in symbols:
        offset = member_offsets[mem_idx]
        body += struct.pack(">II", offset, attrs)

    for sym_name, mem_idx, attrs in symbols:
        body += ascii_to_ebcdic(sym_name) + b"\x00"

    return body


def parse_member_data(raw):
    """Parse the data portion of a --member argument.

    Supports three forms:
      /path/to/file     - read file contents
      hex:<hexstring>   - decode hex bytes
      <ascii string>    - encode as raw ASCII bytes
    """
    if os.path.isfile(raw):
        with open(raw, "rb") as f:
            return f.read()
    if raw.startswith("hex:"):
        return bytes.fromhex(raw[4:])
    return raw.encode("ascii")


# Valid malformation names for --malform-symtab-hdr, mapped to
# make_member_header keyword arguments.
_SYMTAB_HDR_MALFORMATIONS = {
    "bad-terminator": "bad_terminator",
    "empty-name": "empty_name",
    "empty-uid": "empty_uid",
    "empty-gid": "empty_gid",
    "empty-modtime": "empty_modtime",
    "empty-mode": "empty_mode",
}


def build_archive(args):
    """Build the complete archive bytes."""
    output = bytearray()
    output += ZOS_MAGIC

    if args.empty:
        return bytes(output)

    # Parse members.
    members = []
    if args.member:
        for m in args.member:
            parts = m.split(":", 1)
            name = parts[0]
            if len(parts) > 1:
                data = parse_member_data(parts[1])
            else:
                data = b"\x00" * 16  # Dummy content.
            members.append((name, data))

    # Parse symbols.
    symbols = []
    if args.symtab:
        for s in args.symtab:
            parts = s.split(":")
            sym_name = parts[0]
            mem_idx = int(parts[1]) if len(parts) > 1 else 0
            attrs = int(parts[2]) if len(parts) > 2 else 0
            symbols.append((sym_name, mem_idx, attrs))

    # Parse symtab header malformation flags.
    symtab_hdr_kwargs = {}
    if args.malform_symtab_hdr:
        key = args.malform_symtab_hdr
        if key not in _SYMTAB_HDR_MALFORMATIONS:
            sys.exit(
                f"Unknown --malform-symtab-hdr value: {key}. "
                f"Valid: {', '.join(_SYMTAB_HDR_MALFORMATIONS.keys())}"
            )
        symtab_hdr_kwargs[_SYMTAB_HDR_MALFORMATIONS[key]] = True

    # Phase 1: Compute member offsets.
    # Start after magic.
    pos = len(ZOS_MAGIC)

    # If we have a symbol table, it comes first.
    symtab_body = None
    has_symtab = (
        symbols
        or args.symtab_no_symbols
        or args.symtab_truncated
        or args.symtab_bad_count
    )
    if has_symtab:
        # We need to compute the symtab size, but symtab contains member
        # offsets, which depend on symtab size so we do two passes.

        # First pass: compute symtab body with placeholder offsets.
        if args.symtab_truncated:
            symtab_body = make_symtab([], [], truncated=True)
        elif args.symtab_bad_count:
            symtab_body = make_symtab([], [], bad_count=True)
        elif args.symtab_no_symbols:
            symtab_body = struct.pack(">I", 0)  # 0 symbols.
        else:
            placeholder_offsets = [0] * (len(members) + 1)
            symtab_body = make_symtab(symbols, placeholder_offsets)

        symtab_hdr_size = 60  # Fixed header for __.SYMDEF.
        symtab_total = symtab_hdr_size + len(symtab_body)
        # Padding to even boundary.
        if symtab_total % 2 != 0:
            symtab_total += 1
        pos += symtab_total

    # Compute member offsets.
    member_offsets = []
    for name, data in members:
        member_offsets.append(pos)
        hdr_size = 60
        name_ext = 0
        if len(name) > 16:
            name_ext = len(ascii_to_ebcdic(name))
        total = hdr_size + name_ext + len(data)
        if total % 2 != 0:
            total += 1
        pos += total

    # Second pass: recompute symtab with correct offsets.
    if symbols and not args.symtab_truncated and not args.symtab_bad_count:
        symtab_body = make_symtab(symbols, member_offsets)

    # Phase 2: Write output.
    if symtab_body is not None:
        symtab_hdr = make_member_header(
            "__.SYMDEF", 0, 0, 0, 0, len(symtab_body), **symtab_hdr_kwargs
        )
        output += symtab_hdr
        output += symtab_body
        # Pad to even boundary.
        if len(output) % 2 != 0:
            output += EBCDIC_NEWLINE

    for i, (name, data) in enumerate(members):
        hdr = make_member_header(
            name,
            1234567890,
            0,
            0,
            100644,
            len(data),
            bad_terminator=args.bad_terminator,
            empty_name=args.empty_name,
            empty_uid=args.empty_uid,
            empty_gid=args.empty_gid,
            empty_modtime=args.empty_modtime,
            empty_mode=args.empty_mode,
        )
        output += hdr
        output += data
        if len(output) % 2 != 0:
            output += EBCDIC_NEWLINE

    return bytes(output)


def main():
    parser = argparse.ArgumentParser(
        description="Generate z/OS archive files for testing"
    )
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument(
        "--empty", action="store_true", help="Create an empty archive (magic only)"
    )
    parser.add_argument(
        "--member",
        action="append",
        help="Add member as name[:data]. "
        "Data can be a file path, hex:DEADBEEF, "
        "or a raw ASCII string. If omitted, uses "
        "16 zero bytes as dummy content.",
    )
    parser.add_argument(
        "--symtab", action="append", help="Add symbol: name[:member_index[:attributes]]"
    )
    parser.add_argument(
        "--symtab-no-symbols",
        action="store_true",
        help="Add empty symbol table (0 symbols)",
    )
    parser.add_argument(
        "--symtab-truncated", action="store_true", help="Create truncated symbol table"
    )
    parser.add_argument(
        "--symtab-bad-count", action="store_true", help="Symbol count exceeds buffer"
    )
    parser.add_argument(
        "--malform-symtab-hdr",
        metavar="MALFORMATION",
        help="Apply a malformation to the __.SYMDEF header. "
        "Valid values: bad-terminator, empty-name, "
        "empty-uid, empty-gid, empty-modtime, empty-mode",
    )
    parser.add_argument(
        "--bad-terminator",
        action="store_true",
        help="Use invalid terminator on member headers",
    )
    parser.add_argument(
        "--empty-name",
        action="store_true",
        help="Empty/space-leading name on member headers",
    )
    parser.add_argument(
        "--empty-uid", action="store_true", help="Empty UID on member headers"
    )
    parser.add_argument(
        "--empty-gid", action="store_true", help="Empty GID on member headers"
    )
    parser.add_argument(
        "--empty-modtime",
        action="store_true",
        help="Empty LastModified on member headers",
    )
    parser.add_argument(
        "--empty-mode", action="store_true", help="Empty AccessMode on member headers"
    )
    args = parser.parse_args()

    data = build_archive(args)
    with open(args.output, "wb") as f:
        f.write(data)


if __name__ == "__main__":
    main()
