# RUN: llvm-mc --verify -triple riscv32 -mattr=+zdinx %s

# Unsupport Odd Registers in RV32
fadd.d a0, a1, a2 # expected-error:12 {{double precision floating point operands must use even numbered X register}}

# Not support float registers
flw fa4, 12(sp) # expected-error:1 {{instruction requires the following: 'F' (Single-Precision Floating-Point)}}
fadd.d fa0, fa1, fa2 # expected-error:1 {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

# Invalid instructions
fsw a5, 12(sp) # expected-error:1 {{instruction requires the following: 'F' (Single-Precision Floating-Point)}}
fmv.x.w s0, s1 # expected-error:1 {{invalid instruction}}

# Invalid register names
fadd.d a100, a2, a4 # expected-error:8 {{register must be an even-numbered GPR when used as an FP operand}}
fsgnjn.d a100, a2, a4 # expected-error:10 {{register must be an even-numbered GPR when used as an FP operand}}

# Rounding mode when a register is expected
fmadd.d x10, x12, x14, ree # expected-error:24 {{register must be an even-numbered GPR when used as an FP operand}}

# Invalid rounding modes
fmadd.d x10, x12, x14, x16, ree # expected-error:29 {{operand must be a valid floating point rounding mode mnemonic}}
fmsub.d x10, x12, x14, x16, 0 # expected-error:29 {{operand must be a valid floating point rounding mode mnemonic}}
fnmsub.d x10, x12, x14, x16, 0b111 # expected-error:30 {{operand must be a valid floating point rounding mode mnemonic}}

# FP registers where integer regs are expected
fcvt.wu.d ft2, a2 # expected-error:11 {{register must be a GPR}}
