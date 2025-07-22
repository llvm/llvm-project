#!/usr/bin/env python3

import re
import sys

if len(sys.argv) !=2 :
    print("Usage: has_straddle.py <dissasembly file>")
    sys.exit(1)

inputFilename = sys.argv[1]
address_and_encoding_regex = r"// (\S{12}):(( [0-9A-F]{8})+)";

file = open(inputFilename)

for line in file :
    match = re.search(address_and_encoding_regex,line)
    if match :
        hexaddress = match.group(1)
        encoding = match.group(2)
        dwords = encoding.split()
        address = int(hexaddress, 16)
        address_end = address + len(dwords)*4 - 1
        #Cache-line is 64 bytes.  Check for half cache-line straddle.
        if address//32 != address_end//32:
            print("Straddling instruction found at:")
            print(line)
            sys.exit(1)

sys.exit(0)
