# RUN: not llvm-mc -triple=riscv32 -mattr=+zclsd < %s 2>&1 | FileCheck %s

## GPRPairC
c.ld t1, 4(sp)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:6: note: register must be a GPR from x8 to x15
# CHECK: :[[@LINE-3]]:6: note: register pair must start with x8, x10, x12, or x14

c.sd s2, 4(sp)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:6: note: register must be a GPR from x8 to x15
# CHECK: :[[@LINE-3]]:6: note: register pair must start with x8, x10, x12, or x14

## GPRPairNoX0
c.ldsp  x0, 4(sp)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: register must be a GPR excluding zero (x0)
# CHECK: :[[@LINE-3]]:9: note: register pair must start with an even GPR other than x0

c.ldsp  zero, 4(sp)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: register must be a GPR excluding zero (x0)
# CHECK: :[[@LINE-3]]:9: note: register pair must start with an even GPR other than x0

## uimm9_lsb000
c.ldsp t1, 512(sp) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 8 bytes in the range [0, 504]
c.sdsp t1, -8(sp) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 8 bytes in the range [0, 504]

## uimm8_lsb000
c.ld  s0, -8(sp)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:7: note: register must be a GPR from x8 to x15
# CHECK: :[[@LINE-3]]:11: note: invalid operand for instruction
# CHECK: :[[@LINE-4]]:11: note: immediate must be a multiple of 8 bytes in the range [0, 248]

c.sd  s0, 256(sp)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:7: note: register must be a GPR from x8 to x15
# CHECK: :[[@LINE-3]]:11: note: invalid operand for instruction
# CHECK: :[[@LINE-4]]:11: note: immediate must be a multiple of 8 bytes in the range [0, 248]

# Invalid register names
c.ld a1, 4(sp) # CHECK: :[[@LINE]]:6: error: register must be even
c.sd a3, 4(sp) # CHECK: :[[@LINE]]:6: error: register must be even
c.ldsp ra, 4(sp) # CHECK: :[[@LINE]]:8: error: register must be even
c.ldsp t0, 4(sp) # CHECK: :[[@LINE]]:8: error: register must be even
