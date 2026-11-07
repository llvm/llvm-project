# RUN: llvm-mc --verify -triple riscv32 -mattr=+zfinx %s

# Not support float registers
flw fa4, 12(sp) # expected-error:1 {{instruction requires the following: 'F' (Single-Precision Floating-Point)}}
fadd.s fa0, fa1, fa2 # expected-error:1 {{instruction requires the following: 'F' (Single-Precision Floating-Point)}}

# Invalid instructions
fsw a5, 12(sp) # expected-error:1 {{instruction requires the following: 'F' (Single-Precision Floating-Point)}}
fmv.x.w s0, s1 # expected-error:1 {{invalid instruction}}
fadd.d t1, t3, t5 # expected-error:1 {{instruction requires the following: 'Zdinx' (Double in Integer)}}

# Invalid register names
fadd.s a100, a2, a3 # expected-error:8 {{register must be a GPR when used as an FP operand}}
fsgnjn.s a100, a2, a3 # expected-error:10 {{register must be a GPR when used as an FP operand}}

# Rounding mode when a register is expected
fmadd.s x10, x11, x12, ree # expected-error:24 {{register must be a GPR when used as an FP operand}}

# Invalid rounding modes
fmadd.s x10, x11, x12, x13, ree # expected-error:29 {{operand must be a valid floating point rounding mode mnemonic}}
fmsub.s x14, x15, x16, x17, 0 # expected-error:29 {{operand must be a valid floating point rounding mode mnemonic}}
fnmsub.s x18, x19, x20, x21, 0b111 # expected-error:30 {{operand must be a valid floating point rounding mode mnemonic}}

# Using 'Zdinx' instructions for an 'Zfinx'-only target
# odd registers report incorrect registers
fadd.d t0, t1, t2 # expected-error:8 {{double precision floating point operands must use even numbered X register}}
# even registers report the required extension
fadd.d t3, t1, t5 # expected-error:1 {{instruction requires the following: 'Zdinx' (Double in Integer)}}
