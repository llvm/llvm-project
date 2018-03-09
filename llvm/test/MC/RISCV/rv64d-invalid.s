# RUN: not llvm-mc -triple riscv64 -mattr=+d < %s 2>&1 | FileCheck %s

# Integer registers where FP regs are expected
fcvt.l.d ft0, a0 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
fcvt.lu.d ft1, a1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
fmv.x.d ft2, a2 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.d.l a3, ft3 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
fcvt.d.lu a4, ft4 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
fmv.d.x a5, ft5 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
