# RUN: not llvm-mc -triple riscv64 -mattr=+q < %s 2>&1 | FileCheck --match-full-lines %s

# Integer registers where FP regs are expected
fcvt.l.q ft0, a0 # CHECK: <stdin>:[[@LINE]]:10: error: register must be a GPR
fcvt.lu.q ft1, a1 # CHECK: <stdin>:[[@LINE]]:11: error: register must be a GPR

# FP registers where integer regs are expected
fcvt.q.l a3, ft3 # CHECK: <stdin>:[[@LINE]]:10: error: register must be a FPR
fcvt.q.lu a4, ft4 # CHECK: <stdin>:[[@LINE]]:11: error: register must be a FPR
