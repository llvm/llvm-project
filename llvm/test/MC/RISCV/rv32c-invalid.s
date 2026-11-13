# RUN: llvm-mc --verify -triple=riscv32 -mattr=+c < %s
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+zca < %s

## GPRC
.LBB:
c.lw  ra, 4(sp) # expected-error:7 {{register must be a GPR from x8 to x15}}
c.sw  sp, 4(sp) # expected-error:7 {{register must be a GPR from x8 to x15}}
c.beqz  t0, .LBB # expected-error:9 {{register must be a GPR from x8 to x15}}
c.bnez  s8, .LBB # expected-error:9 {{register must be a GPR from x8 to x15}}
c.addi4spn  s4, sp, 12 # expected-error:13 {{register must be a GPR from x8 to x15}}
c.srli  s7, 12 # expected-error:9 {{register must be a GPR from x8 to x15}}
c.srai  t0, 12 # expected-error:9 {{register must be a GPR from x8 to x15}}
c.andi  t1, 12 # expected-error:9 {{register must be a GPR from x8 to x15}}
c.and  t1, a0 # expected-error:8 {{register must be a GPR from x8 to x15}}
c.or   a0, s8 # expected-error:12 {{register must be a GPR from x8 to x15}}
c.xor  t2, a0 # expected-error:8 {{register must be a GPR from x8 to x15}}
c.sub  a0, s8 # expected-error:12 {{register must be a GPR from x8 to x15}}

## GPRNoX0
c.lwsp  x0, 4(sp) # expected-error:9 {{register must be a GPR excluding zero (x0)}}
c.lwsp  zero, 4(sp) # expected-error:9 {{register must be a GPR excluding zero (x0)}}
c.jr  x0 # expected-error:7 {{register must be a GPR excluding zero (x0)}}
c.jalr  zero # expected-error:9 {{register must be a GPR excluding zero (x0)}}
c.mv  ra, x0 # expected-error:11 {{register must be a GPR excluding zero (x0)}}
c.add  ra, ra, x0 # expected-error:16 {{unexpected extra operand for instruction}}

## GPRNoX2
c.lui x2, 4 # expected-error:7 {{register must be a GPR excluding sp (x2)}}

## SP
c.addi4spn  a0, a0, 12 # expected-error:17 {{register must be sp (x2)}}
c.addi16sp  t0, 16 # expected-error:13 {{register must be sp (x2)}}

# Out of range immediates

## uimmlog2xlenn
c.slli t0, 64 # expected-error:12 {{immediate must be an integer in the range [0, 31]}}
c.srli a0, 32 # expected-error:12 {{immediate must be an integer in the range [0, 31]}}

## simm6
c.li t0, 128 # expected-error:10 {{immediate must be an integer in the range [-32, 31]}}
c.li t0, foo # expected-error:10 {{immediate must be an integer in the range [-32, 31]}}
c.li t0, %lo(foo) # expected-error:10 {{immediate must be an integer in the range [-32, 31]}}
c.li t0, %hi(foo) # expected-error:10 {{immediate must be an integer in the range [-32, 31]}}
c.andi a0, -33 # expected-error:12 {{immediate must be an integer in the range [-32, 31]}}
c.andi a0, foo # expected-error:12 {{immediate must be an integer in the range [-32, 31]}}
c.andi a0, %lo(foo) # expected-error:12 {{immediate must be an integer in the range [-32, 31]}}
c.andi a0, %hi(foo) # expected-error:12 {{immediate must be an integer in the range [-32, 31]}}
c.addi t0, -33 # expected-error:12 {{immediate must be an integer in the range [-32, 31]}}
c.addi t0, 32 # expected-error:12 {{immediate must be an integer in the range [-32, 31]}}
c.addi t0, foo # expected-error:12 {{immediate must be an integer in the range [-32, 31]}}
c.addi t0, %lo(foo) # expected-error:12 {{immediate must be an integer in the range [-32, 31]}}
c.addi t0, %hi(foo) # expected-error:12 {{immediate must be an integer in the range [-32, 31]}}

## simm6nonzero
c.nop 32
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:7 {{unexpected extra operand for instruction}}
# expected-note@-3:7 {{immediate must be non-zero in the range [-32, 31]}}

## c_lui_imm
c.lui t0, 0 # expected-error:11 {{immediate must be in [0xfffe0, 0xfffff] or [1, 31]}}
c.lui t0, 32 # expected-error:11 {{immediate must be in [0xfffe0, 0xfffff] or [1, 31]}}
c.lui t0, 0xffffdf # expected-error:11 {{immediate must be in [0xfffe0, 0xfffff] or [1, 31]}}
c.lui t0, 0x1000000 # expected-error:11 {{immediate must be in [0xfffe0, 0xfffff] or [1, 31]}}
c.lui t0, foo # expected-error:11 {{immediate must be in [0xfffe0, 0xfffff] or [1, 31]}}

## uimm8_lsb00_optional
c.lwsp  ra, 256(sp) # expected-error:13 {{immediate must be a multiple of 4 bytes in the range [0, 252]}}
c.swsp  ra, -4(sp) # expected-error:13 {{immediate must be a multiple of 4 bytes in the range [0, 252]}}
## uimm7_lsb00_optional
c.lw  s0, -4(sp) # expected-error:11 {{immediate must be a multiple of 4 bytes in the range [0, 124]}}

c.sw  s0, 128(sp) # expected-error:11 {{immediate must be a multiple of 4 bytes in the range [0, 124]}}

## simm9_lsb0
c.bnez  s1, -258 # expected-error:13 {{immediate must be a multiple of 2 bytes in the range [-256, 254]}}
c.beqz  a0, 256 # expected-error:13 {{immediate must be a multiple of 2 bytes in the range [-256, 254]}}

## simm12_lsb0
c.j 2048 # expected-error:5 {{immediate must be a multiple of 2 bytes in the range [-2048, 2046]}}
c.jal -2050 # expected-error:7 {{immediate must be a multiple of 2 bytes in the range [-2048, 2046]}}

## uimm10_lsb00nonzero
c.addi4spn  a0, sp, 0 # expected-error:21 {{immediate must be a multiple of 4 bytes in the range [4, 1020]}}
c.addi4spn  a0, sp, 1024 # expected-error:21 {{immediate must be a multiple of 4 bytes in the range [4, 1020]}}

## simm10_lsb0000nonzero
c.addi16sp  sp, -528 # expected-error:17 {{immediate must be a multiple of 16 bytes and non-zero in the range [-512, 496]}}
c.addi16sp  sp, 512 # expected-error:17 {{immediate must be a multiple of 16 bytes and non-zero in the range [-512, 496]}}
c.addi16sp  sp, 0 # expected-error:17 {{immediate must be a multiple of 16 bytes and non-zero in the range [-512, 496]}}
