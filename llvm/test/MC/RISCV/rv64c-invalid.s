# RUN: llvm-mc --verify -triple=riscv64 -mattr=+c < %s
# RUN: llvm-mc --verify -triple=riscv64 -mattr=+zca < %s

## GPRC
c.ld ra, 4(sp)
# expected-error@-1:1 {{invalid instruction}}
# expected-note@-2 {{register must be a GPR from x8 to x15}}
# expected-note@-3 {{instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions), RV32I Base Instruction Set}}
c.sd sp, 4(sp)
# expected-error@-1:1 {{invalid instruction}}
# expected-note@-2 {{register must be a GPR from x8 to x15}}
# expected-note@-3 {{instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions), RV32I Base Instruction Set}}
c.addw   a0, a7 # expected-error:14 {{register must be a GPR from x8 to x15}}
c.subw   a0, a6 # expected-error:14 {{register must be a GPR from x8 to x15}}

## GPRNoX0
c.ldsp  x0, 4(sp)
# expected-error@-1:1 {{invalid instruction}}
# expected-note@-2 {{register must be a GPR excluding zero (x0)}}
# expected-note@-3 {{instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions), RV32I Base Instruction Set}}
c.ldsp  zero, 4(sp)
# expected-error@-1:1 {{invalid instruction}}
# expected-note@-2 {{register must be a GPR excluding zero (x0)}}
# expected-note@-3 {{instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions), RV32I Base Instruction Set}}

# Out of range immediates

## uimmlog2xlen
c.slli t0, 64 # expected-error:12 {{immediate must be an integer in the range [0, 63]}}
c.srli a0, -1 # expected-error:12 {{immediate must be an integer in the range [0, 63]}}

## simm6
c.addiw t0, -33 # expected-error:13 {{immediate must be an integer in the range [-32, 31]}}
c.addiw t0, 32 # expected-error:13 {{immediate must be an integer in the range [-32, 31]}}
c.addiw t0, foo # expected-error:13 {{immediate must be an integer in the range [-32, 31]}}
c.addiw t0, %lo(foo) # expected-error:13 {{immediate must be an integer in the range [-32, 31]}}
c.addiw t0, %hi(foo) # expected-error:13 {{immediate must be an integer in the range [-32, 31]}}

## uimm9_lsb000
c.ldsp  ra, 512(sp) # expected-error:13 {{immediate must be a multiple of 8 bytes in the range [0, 504]}}
c.sdsp  ra, -8(sp) # expected-error:13 {{immediate must be a multiple of 8 bytes in the range [0, 504]}}
## uimm8_lsb000
c.ld  s0, -8(sp)
# expected-error@-1:1 {{invalid instruction}}
# expected-note@-2 {{immediate must be a multiple of 8 bytes in the range [0, 248]}}
# expected-note@-3 {{instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions), RV32I Base Instruction Set}}
c.sd  s0, 256(sp)
# expected-error@-1:1 {{invalid instruction}}
# expected-note@-2 {{immediate must be a multiple of 8 bytes in the range [0, 248]}}
# expected-note@-3 {{instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions), RV32I Base Instruction Set}}
