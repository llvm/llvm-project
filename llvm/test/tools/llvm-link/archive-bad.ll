# RUN: cp %S/Inputs/f.ll %t.fg.a
# RUN: not llvm-link %S/Inputs/h.ll %t.fg.a -o %t.linked.bc 2>&1 | FileCheck %s

# RUN: rm -f %t.fg.a
# RUN: rm -f %t.linked.bc

# CHECK: file too small to be an archive
