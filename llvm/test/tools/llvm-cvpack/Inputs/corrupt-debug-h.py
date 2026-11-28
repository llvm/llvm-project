"""Overwrite two consecutive .debug$H hashes of a COFF object, as an older
compiler wrote mismatching hashes for some records.

Usage: corrupt-debug-h.py <in.obj> <out.obj> <first record index>
"""
import struct
import sys

src, dst, index = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = bytearray(open(src, "rb").read())
(count,) = struct.unpack_from("<H", data, 2)
table = 20 + struct.unpack_from("<H", data, 16)[0]
for i in range(count):
    header = table + 40 * i
    if data[header : header + 8].rstrip(b"\0") == b".debug$H":
        size, offset = struct.unpack_from("<II", data, header + 16)
        start = offset + 8 + 8 * index
        assert start + 16 <= offset + size
        data[start : start + 16] = bytes(range(16))
        break
else:
    sys.exit("no .debug$H section")
open(dst, "wb").write(data)
