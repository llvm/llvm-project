# RUN: not llvm-mc -triple riscv64 -mattr=+f < %s 2>&1 | FileCheck %s

# Integer registers where FP regs are expected
fcvt.l.s ft0, a0 # CHECK: :[[@LINE]]:10: error: register must be a GPR
fcvt.lu.s ft1, a1 # CHECK: :[[@LINE]]:11: error: register must be a GPR

# FP registers where integer regs are expected
fcvt.s.l a2, ft2 # CHECK: :[[@LINE]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-1]]:10: note: register must be a FPR
# CHECK: :[[@LINE-2]]:10: note: register must be a GPR when used as an FP operand

fcvt.s.lu a3, ft3 # CHECK: :[[@LINE]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-1]]:11: note: register must be a FPR
# CHECK: :[[@LINE-2]]:11: note: register must be a GPR when used as an FP operand
