#!/usr/bin/env python3
"""Rewrite the Offset of a S_PUB32 record in a llvm-pdbutil pdb2yaml dump.

Used to simulate a PDB whose public symbol records disagree with the
executable's section table (as can happen with real-world PDBs).
"""

import re
import sys

if len(sys.argv) != 4:
    sys.exit("usage: corrupt-public-offset.py <yaml-file> <symbol-name> <new-offset>")

yaml_path, symbol_name, new_offset = sys.argv[1:4]
new_offset = int(new_offset, 0)

with open(yaml_path) as f:
    text = f.read()

pattern = re.compile(
    r"(Offset:\s+)\d+(\s*\n\s*Segment:\s+\d+\s*\n\s*Name:\s+"
    + re.escape(symbol_name)
    + r"\b\s*\n)"
)
text, count = pattern.subn(r"\g<1>" + str(new_offset) + r"\g<2>", text, count=1)
if count != 1:
    sys.exit(f"could not find a S_PUB32 record named '{symbol_name}' in {yaml_path}")

with open(yaml_path, "w") as f:
    f.write(text)
