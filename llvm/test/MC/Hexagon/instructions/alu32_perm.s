# RUN: llvm-mc -triple hexagon -filetype=obj %s -o - | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.1.2 ALU32/PERM

# Combine words in to doublewords
# CHECK: 11 df 95 f3
r17 = combine(r31.h, r21.h)
# CHECK: 11 df b5 f3
r17 = combine(r31.h, r21.l)
# CHECK: 11 df d5 f3
r17 = combine(r31.l, r21.h)
# CHECK: 11 df f5 f3
r17 = combine(r31.l, r21.l)
# CHECK: b0 e2 0f 7c
r17:16 = combine(#21, #31)
# CHECK: b0 e2 3f 73
r17:16 = combine(#21, r31)
# CHECK: f0 e3 15 73
r17:16 = combine(r21, #31)
# CHECK: 10 df 15 f5
r17:16 = combine(r21, r31)

# Mux
# CHECK: f1 c3 75 73
r17 = mux(p3, r21, #31)
# CHECK: b1 c2 ff 73
r17 = mux(p3, #21, r31)
# CHECK: b1 e2 8f 7b
r17 = mux(p3, #21, #31)
# CHECK: 71 df 15 f4
r17 = mux(p3, r21, r31)

# Shift word by 16
# CHECK: 11 c0 15 70
r17 = aslh(r21)
# CHECK: 11 c0 35 70
r17 = asrh(r21)

# Pack high and low halfwords
# CHECK: 10 df 95 f5
r17:16 = packhl(r21, r31)
