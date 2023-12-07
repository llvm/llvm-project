# RUN: llvm-as %S/Inputs/f.ll -o %t.f.bc
# RUN: llvm-as %S/Inputs/g.ll -o %t.g.bc
# RUN: llvm-ar cr %t.fg.a %t.f.bc %t.g.bc
# RUN: llvm-ar cr --format=gnu %t.empty.lib
# RUN: llvm-link %S/Inputs/h.ll %t.fg.a %t.empty.lib -o %t.linked.bc

# RUN: llvm-nm %t.linked.bc | FileCheck %s

# RUN: rm -f %t.f.bc
# RUN: rm -f %t.g.bc
# RUN: rm -f %t.fg.a
# RUN: rm -f %t.empty.a
# RUN: rm -f %t.linked.bc

# CHECK: -------- T f
# CHECK: -------- T g
# CHECK: -------- T h
