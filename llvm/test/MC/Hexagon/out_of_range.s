# RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2>&1 | FileCheck %s

r1:0=##0xFFFFFF7000001000
# CHECK: rror: value -144(0xffffffffffffff70) out of range: -128-127

p0 = cmpb.eq(r0, #-257)
# CHECK: rror: invalid operand for instruction

p0 = cmpb.eq(r0, #256)
# CHECK: rror: invalid operand for instruction
