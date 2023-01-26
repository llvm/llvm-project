# RUN: echo -e '!<arch>\nwith invalid contents' > %t.fg.a
# RUN: not llvm-link %S/Inputs/h.ll %t.fg.a -o %t.linked.bc 2>&1 | FileCheck %s

# RUN: rm -f %t.fg.a
# RUN: rm -f %t.linked.bc

# CHECK: truncated or malformed archive
