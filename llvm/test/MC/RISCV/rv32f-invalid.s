# RUN: llvm-mc --verify -triple riscv32 -mattr=+f < %s

# Out of range immediates
## simm12
flw ft1, -2049(a0) # expected-error:10 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
fsw ft2, 2048(a1) # expected-error:10 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}

# Memory operand not formatted correctly
flw ft1, a0, -200 # expected-error:14 {{register must be a GPR excluding zero (x0)}}
fsw ft2, a1, 100 # expected-error:14 {{register must be a GPR excluding zero (x0)}}
flw ft1, a_symbol, x0 # expected-error:20 {{register must be a GPR excluding zero (x0)}}
fsw ft2, a_symbol, x0 # expected-error:20 {{register must be a GPR excluding zero (x0)}}

# Invalid register names
flw ft15, 100(a0) # expected-error:5 {{register must be a FPR}}
flw ft1, 100(a10) # expected-error:14 {{expected register}}
fsgnjn.s fa100, fa2, fa3 # expected-error:10 {{register must be a FPR}}

# Integer registers where FP regs are expected
fmv.x.w fs7, a2 # expected-error:9 {{register must be a GPR}}

# FP registers where integer regs are expected
fmv.w.x a8, ft2 # expected-error:9 {{register must be a FPR}}

# Rounding mode when a register is expected
fmadd.s f10, f11, f12, ree # expected-error:24 {{register must be a FPR}}

# Invalid rounding modes
fmadd.s f10, f11, f12, f13, ree # expected-error:29 {{operand must be a valid floating point rounding mode mnemonic}}
fmsub.s f14, f15, f16, f17, 0 # expected-error:29 {{operand must be a valid floating point rounding mode mnemonic}}
fnmsub.s f18, f19, f20, f21, 0b111 # expected-error:30 {{operand must be a valid floating point rounding mode mnemonic}}

# Using 'D' instructions for an 'F'-only target
fadd.d ft0, ft1, ft2 # expected-error:1 {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

# Using RV64F instructions for RV32 is tested in rv64f-valid.s
