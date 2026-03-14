# RUN: llvm-mc -triple hexagon -filetype=obj %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.1.1 ALU32/ALU

# Add
# CHECK: f1 c3 15 b0
r17 = add(r21, #31)
# CHECK: 11 df 15 f3
r17 = add(r21, r31)
# CHECK: 11 df 55 f6
r17 = add(r21, r31):sat

# And
# CHECK: f1 c3 15 76
r17 = and(r21, #31)
# CHECK: f1 c3 95 76
r17 = or(r21, #31)
# CHECK: 11 df 15 f1
r17 = and(r21, r31)
# CHECK: 11 df 35 f1
r17 = or(r21, r31)
# CHECK: 11 df 75 f1
r17 = xor(r21, r31)
# CHECK: 11 d5 9f f1
r17 = and(r21, ~r31)
# CHECK: 11 d5 bf f1
r17 = or(r21, ~r31)

# Nop
# CHECK: 00 c0 00 7f
nop

# Subtract
# CHECK: b1 c2 5f 76
r17 = sub(#21, r31)
# CHECK: 11 df 35 f3
r17 = sub(r31, r21)
# CHECK: 11 df d5 f6
r17 = sub(r31, r21):sat

# Sign extend
# CHECK: 11 c0 bf 70
r17 = sxtb(r31)

# Transfer immediate
# CHECK: 15 c0 31 72
r17.h = #21
# CHECK: 15 c0 31 71
r17.l = #21
# CHECK: f1 ff 5f 78
r17 = #32767
# CHECK: f1 ff df 78
r17 = #-1

# Transfer register
# CHECK: 11 c0 75 70
r17 = r21

# Vector add halfwords
# CHECK: 11 df 15 f6
r17 = vaddh(r21, r31)
# CHECK: 11 df 35 f6
r17 = vaddh(r21, r31):sat
# CHECK: 11 df 75 f6
r17 = vadduh(r21, r31):sat

# Vector average halfwords
# CHECK: 11 df 15 f7
r17 = vavgh(r21, r31)
# CHECK: 11 df 35 f7
r17 = vavgh(r21, r31):rnd
# CHECK: 11 df 75 f7
r17 = vnavgh(r31, r21)

# Vector subtract halfwords
# CHECK: 11 df 95 f6
r17 = vsubh(r31, r21)
# CHECK: 11 df b5 f6
r17 = vsubh(r31, r21):sat
# CHECK: 11 df f5 f6
r17 = vsubuh(r31, r21):sat

# Zero extend
# CHECK: 11 c0 d5 70
r17 = zxth(r21)
