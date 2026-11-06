# RUN: llvm-mc --verify -triple riscv32 -mattr=+zfbfmin,+d < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+zfbfmin,+d < %s

# Out of range immediates
## simm12
flh ft1, -2049(a0) # expected-error:10 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
fsh ft2, 2048(a1) # expected-error:10 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}

# Memory operand not formatted correctly
flh ft1, a0, -200 # expected-error:14 {{register must be a GPR}}

# Invalid register names
flh ft15, 100(a0) # expected-error:5 {{register must be a FPR}}
flh ft1, 100(a10) # expected-error:14 {{expected register}}

# Integer registers where FP regs are expected
fmv.x.h fs7, a2 # expected-error:9 {{register must be a GPR}}

# FP registers where integer regs are expected
fmv.h.x a8, ft2 # expected-error:9 {{register must be a FPR}}

# Attempting to use fcvt instructions from zfhmin
fcvt.s.h fa0, ft0 # expected-error:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)}}
fcvt.h.s ft2, fa2 # expected-error:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)}}
fcvt.d.h fa0, ft0 # expected-error:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)}}
fcvt.h.d ft2, fa2 # expected-error:1 {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal)}}
