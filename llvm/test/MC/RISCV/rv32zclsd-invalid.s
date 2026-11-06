# RUN: llvm-mc --verify -triple=riscv32 -mattr=+zclsd < %s

## GPRPairC
c.ld t1, 4(sp)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:6 {{register pair must start with x8, x10, x12, or x14}}

c.sd s2, 4(sp)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:6 {{register pair must start with x8, x10, x12, or x14}}

## GPRPairNoX0
c.ldsp  x0, 4(sp)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:9 {{register pair must start with an even GPR other than x0}}

c.ldsp  zero, 4(sp)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:9 {{register pair must start with an even GPR other than x0}}

## uimm9_lsb000
c.ldsp t1, 512(sp) # expected-error:12 {{immediate must be a multiple of 8 bytes in the range [0, 504]}}
c.sdsp t1, -8(sp) # expected-error:12 {{immediate must be a multiple of 8 bytes in the range [0, 504]}}

## uimm8_lsb000
c.ld  s0, -8(sp)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:11 {{immediate must be a multiple of 8 bytes in the range [0, 248]}}

c.sd  s0, 256(sp)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:11 {{immediate must be a multiple of 8 bytes in the range [0, 248]}}

# Invalid register names
c.ld a1, 4(sp) # expected-error:6 {{register must be even}}
c.sd a3, 4(sp) # expected-error:6 {{register must be even}}
c.ldsp ra, 4(sp) # expected-error:8 {{register must be even}}
c.ldsp t0, 4(sp) # expected-error:8 {{register must be even}}
