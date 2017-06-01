#!/usr/bin/env python
#
# Mark functions in an arm assembly file. This is done by surrounding the
# function with "# -- Begin Name" and "# -- End Name"
# (This script is designed for arm ios assembly syntax)
import sys
import re

inp = open(sys.argv[1], "r").readlines()

# First pass
linenum = 0
INVALID=-100
last_align = INVALID
last_code = INVALID
last_globl = INVALID
begin = INVALID
begins = dict()
for line in inp:
    linenum += 1
    if ".align" in line:
        last_align = linenum
    if ".code" in line:
        last_code = linenum
    if ".globl" in line:
        last_globl = linenum
    m = re.search(r'.thumb_func\s+(\w+)', line)
    if m:
        funcname = m.group(1)
        if last_code == last_align+1 and (linenum - last_code) < 4:
            begin = last_align
            if last_globl+1 == last_align:
                begin = last_globl
    if line == "\n" and begin != INVALID:
        end = linenum
        triple = (funcname, begin, end)
        begins[begin] = triple
        begin = INVALID

# Second pass: Mark
out = open(sys.argv[1], "w")
in_func = None
linenum = 0
for line in inp:
    linenum += 1
    if in_func is not None and linenum == end:
        out.write("# -- End  %s\n" % in_func)
        in_func = None

    triple = begins.get(linenum)
    if triple is not None:
        in_func, begin, end = triple
        out.write("# -- Begin  %s\n" % in_func)
    out.write(line)
