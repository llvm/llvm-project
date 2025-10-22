# RUN: not llvm-mc -triple=riscv32 -mattr=+zclsd < %s 2>&1 | FileCheck %s

## GPRPairC
c.ld t1, 4(sp) # CHECK: :[[@LINE]]:6: error: invalid operand for instruction
c.sd s2, 4(sp) # CHECK: :[[@LINE]]:6: error: invalid operand for instruction

## GPRPairNoX0
c.ldsp  x0, 4(sp) # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
c.ldsp  zero, 4(sp) # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

## uimm9_lsb000
c.ldsp t1, 512(sp) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 8 bytes in the range [0, 504]
c.sdsp t1, -8(sp) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 8 bytes in the range [0, 504]
## uimm8_lsb000
c.ld  s0, -8(sp) # CHECK: :[[@LINE]]:11: error: immediate must be a multiple of 8 bytes in the range [0, 248]
c.sd  s0, 256(sp) # CHECK: :[[@LINE]]:11: error: immediate must be a multiple of 8 bytes in the range [0, 248]

# Invalid register names
c.ld a1, 4(sp) # CHECK: :[[@LINE]]:6: error: register must be even
c.sd a3, 4(sp) # CHECK: :[[@LINE]]:6: error: register must be even
c.ldsp ra, 4(sp) # CHECK: :[[@LINE]]:8: error: register must be even
c.ldsp t0, 4(sp) # CHECK: :[[@LINE]]:8: error: register must be even
