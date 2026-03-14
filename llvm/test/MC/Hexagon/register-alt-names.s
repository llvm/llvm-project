# RUN: llvm-mc -triple=hexagon < %s | FileCheck %s

# CHECK: r0 = r31
r0 = lr

# CHECK: r1 = r30
r1 = fp

# CHECK: r2 = r29
r2 = sp

# CHECK: r1:0 = combine(r31,r30)
r1:0 = lr:fp

