# RUN: llvm-mc --verify -triple riscv64 -mattr=+q < %s

# Integer registers where FP regs are expected
fcvt.l.q ft0, a0 # expected-error:10 {{register must be a GPR}}
fcvt.lu.q ft1, a1 # expected-error:11 {{register must be a GPR}}

# FP registers where integer regs are expected
fcvt.q.l a3, ft3 # expected-error:10 {{register must be a FPR}}
fcvt.q.lu a4, ft4 # expected-error:11 {{register must be a FPR}}
