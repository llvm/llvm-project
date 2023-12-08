# RUN: not llvm-mc -triple riscv64 -mattr=+f < %s 2>&1 | FileCheck %s

# Integer registers where FP regs are expected
fcvt.l.s ft0, a0 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
fcvt.lu.s ft1, a1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.s.l a2, ft2 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
fcvt.s.lu a3, ft3 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
