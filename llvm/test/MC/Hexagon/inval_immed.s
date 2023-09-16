# RUN: not llvm-mc -filetype=asm -triple=hexagon %s 2>%t; FileCheck %s < %t

    .text
r0 = mpyi(r0,#m9)

# CHECK: error: invalid operand for instruction
