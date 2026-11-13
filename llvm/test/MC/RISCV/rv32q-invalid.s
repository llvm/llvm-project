# RUN: llvm-mc --verify -triple riscv32 -mattr=+q < %s

# Out of range immediates
## simm12
flq ft1, -2049(a0) # expected-error:10 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
fsq ft2, 2048(a1) # expected-error:10 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}

# Memory operand not formatted correctly
flq ft1, a0, -200 # expected-error:14 {{register must be a GPR excluding zero (x0)}}
fsq ft2, a1, 100 # expected-error:14 {{register must be a GPR excluding zero (x0)}}
flq ft1, a_symbol, x0 # expected-error:20 {{register must be a GPR excluding zero (x0)}}
fsq ft2, a_symbol, x0 # expected-error:20 {{register must be a GPR excluding zero (x0)}}

# Invalid register names
flq ft15, 100(a0) # expected-error:5 {{register must be a FPR}}
flq ft1, 100(a10) # expected-error:14 {{expected register}}
fsgnjn.q fa100, fa2, fa3 # expected-error:10 {{register must be a FPR}}

# Integer registers where FP regs are expected
fadd.q a2, a1, a0 # expected-error:8 {{register must be a FPR}}

# FP registers where integer regs are expected
fcvt.wu.q ft2, a1 # expected-error:11 {{register must be a GPR}}
