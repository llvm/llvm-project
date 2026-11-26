# RUN: llvm-mc --verify -triple riscv32 -mattr=+a,+zacas < %s

# Non-zero offsets not supported for the third operand (rs1).
amocas.w a1, a3, 1(a5) # expected-error:18 {{optional integer offset must be 0}}
amocas.d a0, a2, 2(a5) # expected-error:18 {{optional integer offset must be 0}}

# First and second operands (rd and rs2) of amocas.d must be even for RV32.
amocas.d a1, a2, (a1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:10 {{register must be even}}
amocas.d a0, a1, (a1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:14 {{register must be even}}
amocas.d.aq a1, a2, (a1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:13 {{register must be even}}
amocas.d.aq a0, a1, (a1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:17 {{register must be even}}
amocas.d.rl a1, a2, (a1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:13 {{register must be even}}
amocas.d.rl a0, a1, (a1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:17 {{register must be even}}
amocas.d.aqrl a1, a2, (a1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:15 {{register must be even}}
amocas.d.aqrl a0, a1, (a1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# expected-note@-3:19 {{register must be even}}

# amocas.q is not supported for RV32.
amocas.q a0, a0, (a1) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
