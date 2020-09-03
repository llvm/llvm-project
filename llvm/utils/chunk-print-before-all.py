#!/usr/bin/env python
# Given a -print-before-all -print-module-scope log from an opt invocation,
# chunk it into a series of individual IR files, one for each pass invocation.
# If the log ends with an obvious stack trace, try to split off a separate
# "crashinfo.txt" file leaving only the valid input IR in the last chunk.
# Files are written to current working directory.

from __future__ import print_function

import sys

basename = "chunk-"
chunk_id = 0

def print_chunk(lines):
    global chunk_id
    global basename
    fname = basename + str(chunk_id) + ".ll"
    chunk_id = chunk_id + 1
    print("writing chunk " + fname + " (" + str(len(lines)) + " lines)")
    with open(fname, "w") as f:
        f.writelines(lines)

is_dump = False
cur = []
for line in sys.stdin:
    if line.startswith("*** IR Dump Before "):
        if len(cur) != 0:
            print_chunk(cur);
            cur = []
        cur.append("; " + line)
    elif line.startswith("Stack dump:"):
        print_chunk(cur);
        cur = []
        cur.append(line)
        is_dump = True
    else:
        cur.append(line)

if is_dump:
    print("writing crashinfo.txt (" + str(len(cur)) + " lines)")
    with open("crashinfo.txt", "w") as f:
        f.writelines(cur)
else:
    print_chunk(cur);
