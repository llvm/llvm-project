# RUN: llvm-mc --verify -triple riscv32 -mattr=+zfh < %s

# Out of range immediates
## simm12
flh ft1, -2049(a0) # expected-error:10 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
fsh ft2, 2048(a1) # expected-error:10 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}

# Memory operand not formatted correctly
flh ft1, a0, -200 # expected-error:14 {{register must be a GPR}}
fsw ft2, a1, 100 # expected-error:14 {{register must be a GPR}}

# Invalid register names
flh ft15, 100(a0) # expected-error:5 {{register must be a FPR}}
flh ft1, 100(a10) # expected-error:14 {{expected register}}
fsgnjn.h fa100, fa2, fa3 # expected-error:10 {{register must be a FPR}}

# Integer registers where FP regs are expected
fmv.x.h fs7, a2 # expected-error:9 {{register must be a GPR}}

# FP registers where integer regs are expected
fmv.h.x a8, ft2 # expected-error:9 {{register must be a FPR}}

# Rounding mode when a register is expected
fmadd.h f10, f11, f12, ree # expected-error:24 {{register must be a FPR}}

# Invalid rounding modes
fmadd.h f10, f11, f12, f13, ree # expected-error:29 {{operand must be a valid floating point rounding mode mnemonic}}
fmsub.h f14, f15, f16, f17, 0 # expected-error:29 {{operand must be a valid floating point rounding mode mnemonic}}
fnmsub.h f18, f19, f20, f21, 0b111 # expected-error:30 {{operand must be a valid floating point rounding mode mnemonic}}

# Integer registers where FP regs are expected
fadd.h a2, a1, a0 # expected-error:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-1:8 {{register must be a FPR}}
# expected-note@-2:1 {{instruction requires the following: 'Zhinx' (Half Float in Integer)}}

# FP registers where integer regs are expected
fcvt.wu.h ft2, a1 # expected-error:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-1:11 {{register must be a GPR}}
# expected-note@-2:1 {{instruction requires the following: 'Zhinx' (Half Float in Integer)}}
