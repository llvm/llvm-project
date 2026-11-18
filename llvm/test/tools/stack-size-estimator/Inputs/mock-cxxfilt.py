#!/usr/bin/env python3
import sys

for line in sys.stdin:
    name = line.strip()
    if name.startswith("_Z"):
        print(f"demangled_{name[2:]}")
    else:
        print(name)
    sys.stdout.flush()
