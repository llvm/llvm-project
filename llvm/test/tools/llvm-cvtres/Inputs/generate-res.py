"""Generate a .res file with many resources for the llvm-cvtres tests.

The file contains one resource for each combination of TYPES numeric types and
NAMES numeric names, both counted from 1.
Every resource has the language 1033 (en-US) and four bytes of data.
"""

import argparse
import struct

LANG_EN_US = 0x409
MEMORY_FLAGS = 0x1030  # MOVEABLE | PURE | DISCARDABLE, the rc.exe default.

# Every .res file starts with an empty entry with the numeric type and name 0.
NULL_ENTRY = struct.pack("<IIHHHH", 0, 32, 0xFFFF, 0, 0xFFFF, 0) + bytes(16)


def encode_id(value):
    # A numeric type or name is stored as 0xffff followed by the ID.
    return struct.pack("<HH", 0xFFFF, value)


def encode_entry(type_id, name, data):
    # The header starts with the size of the data and its own size,
    # followed by the type, the name and a fixed-size part.
    header = encode_id(type_id) + encode_id(name)
    header += struct.pack("<IHHII", 0, MEMORY_FLAGS, LANG_EN_US, 0, 0)
    sizes = struct.pack("<II", len(data), 8 + len(header))
    # The data is padded to a multiple of four bytes.
    return sizes + header + data + bytes(-len(data) % 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--types", type=int, default=1)
    parser.add_argument("--names", type=int, default=1)
    parser.add_argument("output")
    args = parser.parse_args()

    entries = [NULL_ENTRY]
    for type_id in range(1, args.types + 1):
        for name in range(1, args.names + 1):
            entries.append(encode_entry(type_id, name, b"DATA"))
    with open(args.output, "wb") as f:
        f.write(b"".join(entries))


if __name__ == "__main__":
    main()
