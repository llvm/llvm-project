# RUN: not llvm-mc -triple=riscv64 -mattr=+c < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+zca < %s 2>&1 | FileCheck %s

## GPRC
c.ld ra, 4(sp) # CHECK: :[[@LINE]]:6: error: invalid operand for instruction
c.sd sp, 4(sp) # CHECK: :[[@LINE]]:6: error: invalid operand for instruction
c.addw   a0, a7 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
c.subw   a0, a6 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction

## GPRNoX0
c.ldsp  x0, 4(sp) # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
c.ldsp  zero, 4(sp) # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

# Out of range immediates

## uimmlog2xlennonzero
c.slli t0, 64 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [1, 63]
c.srli a0, -1 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [1, 63]
c.srai a0, 0  # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [1, 63]

## simm6
c.addiw t0, -33 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [-32, 31]
c.addiw t0, 32 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [-32, 31]
c.addiw t0, foo # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [-32, 31]
c.addiw t0, %lo(foo) # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [-32, 31]
c.addiw t0, %hi(foo) # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [-32, 31]

## uimm9_lsb000
c.ldsp  ra, 512(sp) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 8 bytes in the range [0, 504]
c.sdsp  ra, -8(sp) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 8 bytes in the range [0, 504]
## uimm8_lsb000
c.ld  s0, -8(sp) # CHECK: :[[@LINE]]:11: error: immediate must be a multiple of 8 bytes in the range [0, 248]
c.sd  s0, 256(sp) # CHECK: :[[@LINE]]:11: error: immediate must be a multiple of 8 bytes in the range [0, 248]
