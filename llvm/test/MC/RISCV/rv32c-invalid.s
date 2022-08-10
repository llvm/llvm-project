# RUN: not llvm-mc -triple=riscv32 -mattr=+c -mattr=+no-rvc-hints < %s 2>&1 \
# RUN:     | FileCheck %s

## GPRC
.LBB:
c.lw  ra, 4(sp) # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
c.sw  sp, 4(sp) # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
c.beqz  t0, .LBB # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
c.bnez  s8, .LBB # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
c.addi4spn  s4, sp, 12 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
c.srli  s7, 12 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
c.srai  t0, 12 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
c.andi  t1, 12 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
c.and  t1, a0 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
c.or   a0, s8 # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
c.xor  t2, a0 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
c.sub  a0, s8 # CHECK: :[[@LINE]]:12: error: invalid operand for instruction

## GPRNoX0
c.lwsp  x0, 4(sp) # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
c.lwsp  zero, 4(sp) # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
c.jr  x0 # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
c.jalr  zero # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
c.addi  x0, x0, 1 # CHECK: :[[@LINE]]:13: error: immediate must be zero
c.li  zero, 2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RVC Hint Instructions{{$}}
c.slli  zero, zero, 4 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
c.mv  zero, s0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RVC Hint Instructions{{$}}
c.mv  ra, x0 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
c.add  ra, ra, x0 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
c.add  zero, zero, sp # CHECK: :[[@LINE]]:14: error: invalid operand for instruction

## GPRNoX0X2
c.lui x0, 4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RVC Hint Instructions{{$}}
c.lui x2, 4 # CHECK: :[[@LINE]]:7: error: invalid operand for instruction

## SP
c.addi4spn  a0, a0, 12 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
c.addi16sp  t0, 16 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction

# Out of range immediates

## uimmlog2xlennonzero
c.slli t0, 64 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [1, 31]
c.srli a0, 32 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [1, 31]
c.srai a0, 0  # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [1, 31]

## simm6
c.li t0, 128 # CHECK: :[[@LINE]]:10: error: immediate must be an integer in the range [-32, 31]
c.li t0, foo # CHECK: :[[@LINE]]:10: error: immediate must be an integer in the range [-32, 31]
c.li t0, %lo(foo) # CHECK: :[[@LINE]]:10: error: immediate must be an integer in the range [-32, 31]
c.li t0, %hi(foo) # CHECK: :[[@LINE]]:10: error: immediate must be an integer in the range [-32, 31]
c.andi a0, -33 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [-32, 31]
c.andi a0, foo # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [-32, 31]
c.andi a0, %lo(foo) # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [-32, 31]
c.andi a0, %hi(foo) # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [-32, 31]

## simm6nonzero
c.addi t0, 0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RVC Hint Instructions{{$}}
c.addi t0, -33 # CHECK: :[[@LINE]]:12: error: immediate must be non-zero in the range [-32, 31]
c.addi t0, 32 # CHECK: :[[@LINE]]:12: error: immediate must be non-zero in the range [-32, 31]
c.addi t0, foo # CHECK: :[[@LINE]]:12: error: immediate must be non-zero in the range [-32, 31]
c.addi t0, %lo(foo) # CHECK: :[[@LINE]]:12: error: immediate must be non-zero in the range [-32, 31]
c.addi t0, %hi(foo) # CHECK: :[[@LINE]]:12: error: immediate must be non-zero in the range [-32, 31]

## c_lui_imm
c.lui t0, 0 # CHECK: :[[@LINE]]:11: error: immediate must be in [0xfffe0, 0xfffff] or [1, 31]
c.lui t0, 32 # CHECK: :[[@LINE]]:11: error: immediate must be in [0xfffe0, 0xfffff] or [1, 31]
c.lui t0, 0xffffdf # CHECK: :[[@LINE]]:11: error: immediate must be in [0xfffe0, 0xfffff] or [1, 31]
c.lui t0, 0x1000000 # CHECK: :[[@LINE]]:11: error: immediate must be in [0xfffe0, 0xfffff] or [1, 31]

## uimm8_lsb00
c.lwsp  ra, 256(sp) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 4 bytes in the range [0, 252]
c.swsp  ra, -4(sp) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 4 bytes in the range [0, 252]
## uimm7_lsb00
c.lw  s0, -4(sp) # CHECK: :[[@LINE]]:11: error: immediate must be a multiple of 4 bytes in the range [0, 124]
c.sw  s0, 128(sp) # CHECK: :[[@LINE]]:11: error: immediate must be a multiple of 4 bytes in the range [0, 124]

## simm9_lsb0
c.bnez  s1, -258 # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 2 bytes in the range [-256, 254]
c.beqz  a0, 256 # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 2 bytes in the range [-256, 254]

## simm12_lsb0
c.j 2048 # CHECK: :[[@LINE]]:5: error: immediate must be a multiple of 2 bytes in the range [-2048, 2046]
c.jal -2050 # CHECK: :[[@LINE]]:7: error: immediate must be a multiple of 2 bytes in the range [-2048, 2046]

## uimm10_lsb00nonzero
c.addi4spn  a0, sp, 0 # CHECK: :[[@LINE]]:21: error: immediate must be a multiple of 4 bytes in the range [4, 1020]
c.addi4spn  a0, sp, 1024 # CHECK: :[[@LINE]]:21: error: immediate must be a multiple of 4 bytes in the range [4, 1020]

## simm10_lsb0000nonzero
c.addi16sp  sp, -528 # CHECK: :[[@LINE]]:17: error: immediate must be a multiple of 16 bytes and non-zero in the range [-512, 496]
c.addi16sp  sp, 512 # CHECK: :[[@LINE]]:17: error: immediate must be a multiple of 16 bytes and non-zero in the range [-512, 496]
c.addi16sp  sp, 0 # CHECK: :[[@LINE]]:17: error: immediate must be a multiple of 16 bytes and non-zero in the range [-512, 496]
