# RUN: llvm-mc --verify -triple riscv64 -mattr=+zdinx %s

# Not support float registers
fld fa4, 12(sp) # expected-error:1 {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}
ld a0, -2049(a1) # expected-error:8 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}

# Invalid instructions
fsd a5, 12(sp) # expected-error:1 {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}
sd fa4, 64(sp) # expected-error:4 {{register must be a GPR}}
fmv.x.d t2, a2 # expected-error:1 {{invalid instruction}}
fmv.d.x a5, t5 # expected-error:1 {{invalid instruction}}

# FP registers where integer regs are expected
fcvt.d.l a3, ft3 # expected-error:14 {{register must be a GPR}}
fcvt.d.lu a4, ft4 # expected-error:15 {{register must be a GPR}}
