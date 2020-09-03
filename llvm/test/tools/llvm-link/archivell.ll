# RUN: llvm-ar cr %t.fg.a %S/Inputs/f.ll %S/Inputs/g.ll
# RUN: not llvm-link %S/Inputs/h.ll %t.fg.a -o %t.linked.bc 2>&1 | FileCheck %s

# RUN: rm -f %t.fg.a
# RUN: rm -f %t.linked.bc

# CHECK: error: member of archive is not a bitcode file
