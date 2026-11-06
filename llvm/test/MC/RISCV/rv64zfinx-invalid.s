# RUN: llvm-mc --verify -triple riscv64 -mattr=+zfinx %s

# Not support float registers
flw fa4, 12(sp) # expected-error:1 {{instruction requires the following: 'F' (Single-Precision Floating-Point)}}

# Invalid instructions
fsw a5, 12(sp) # expected-error:1 {{instruction requires the following: 'F' (Single-Precision Floating-Point)}}
fmv.x.w t2, a2 # expected-error:1 {{invalid instruction}}
fmv.w.x a5, t5 # expected-error:1 {{invalid instruction}}

# FP registers where integer regs are expected
fcvt.s.l a2, ft2 # expected-error:14 {{register must be a GPR}}
fcvt.s.lu a3, ft3 # expected-error:15 {{register must be a GPR}}
