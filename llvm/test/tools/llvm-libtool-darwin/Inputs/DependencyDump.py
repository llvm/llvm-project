#
# Dump the dependency file (produced with -dependency_info) to text
# format for testing purposes.
#

import sys

f = open(sys.argv[1], "rb")
byte = f.read(1)
while byte != b"":
    if byte == b"\x00":
        sys.stdout.write("version: ")
    elif byte == b"\x10":
        sys.stdout.write("input-file: ")
    elif byte == b"\x11":
        sys.stdout.write("not-found: ")
    elif byte == b"\x40":
        sys.stdout.write("output-file: ")
    byte = f.read(1)
    while byte != b"\x00":
        sys.stdout.write(byte.decode("ascii"))
        byte = f.read(1)
    sys.stdout.write("\n")
    byte = f.read(1)

f.close()
