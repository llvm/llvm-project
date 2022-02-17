# RUN: not llvm-mc -triple riscv64 -mattr=+zfinx %s 2>&1 | FileCheck %s

# Invalid instructions
fmv.x.w t2, a2 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
fmv.w.x a5, t5 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.s.l a2, ft2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
fcvt.s.lu a3, ft3 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
