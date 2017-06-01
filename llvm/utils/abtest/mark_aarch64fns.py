#!/usr/bin/env python
#
# Mark functions in an arm assembly file. This is done by surrounding the
# function with "# -- Begin Name" and "# -- End Name"
# (This script is designed for aarch64 ios assembly syntax)
import sys
import re

inp = open(sys.argv[1], "r").readlines()

# First pass
linenum = 0
INVALID=-100
last_align = INVALID
last_code = INVALID
last_globl = INVALID
last_globl_name = None
begin = INVALID
in_text_section = False
begins = dict()
for line in inp:
    linenum += 1
    if re.search(r'.section\s+__TEXT,__text,regular,pure_instructions', line):
        in_text_section = True
        continue
    elif ".section" in line:
        in_text_section = False
        continue

    if not in_text_section:
        continue

    if ".align" in line:
        last_align = linenum
    gl = re.search(r'.globl\s+(\w+)', line)
    if gl:
        last_globl_name = gl.group(1)
        last_globl = linenum
    m = re.search(r'^(\w+):', line)
    if m and begin == INVALID:
        labelname = m.group(1)
        if last_globl+2 == linenum and last_globl_name == labelname:
            begin = last_globl
            funcname = labelname
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
