# RUN: llvm-mc --verify -triple riscv32 -mattr=+zhinx %s

# Not support float registers
flw fa4, 12(sp) # expected-error:1 {{instruction requires the following: 'F' (Single-Precision Floating-Point)}}
fadd.h fa0, fa1, fa2 # expected-error:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}

# Invalid instructions
fsw a5, 12(sp) # expected-error:1 {{instruction requires the following: 'F' (Single-Precision Floating-Point)}}
fmv.x.h s0, s1 # expected-error:1 {{invalid instruction}}

# Invalid register names
fadd.h a100, a2, a3 # expected-error:8 {{register must be a GPR when used as an FP operand}}
fsgnjn.h a100, a2, a3 # expected-error:10 {{register must be a GPR when used as an FP operand}}

# Rounding mode when a register is expected
fmadd.h x10, x11, x12, ree # expected-error:24 {{register must be a GPR when used as an FP operand}}

# Invalid rounding modes
fmadd.h x10, x11, x12, x13, ree # expected-error:29 {{operand must be a valid floating point rounding mode mnemonic}}
fmsub.h x14, x15, x16, x17, 0 # expected-error:29 {{operand must be a valid floating point rounding mode mnemonic}}
fnmsub.h x18, x19, x20, x21, 0b111 # expected-error:30 {{operand must be a valid floating point rounding mode mnemonic}}

# FP registers where integer regs are expected
fcvt.wu.h ft2, a1 # expected-error:11 {{register must be a GPR}}
