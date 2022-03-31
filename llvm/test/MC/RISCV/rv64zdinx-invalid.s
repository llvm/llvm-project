# RUN: not llvm-mc -triple riscv64 -mattr=+zdinx %s 2>&1 | FileCheck %s

# Invalid Instructions
fmv.x.d t2, a2 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
fmv.d.x a5, t5 # CHECK: :[[@LINE]]:9:  error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.d.l a3, ft3 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
fcvt.d.lu a4, ft4 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
