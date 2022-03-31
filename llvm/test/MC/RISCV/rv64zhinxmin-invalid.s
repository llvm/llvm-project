# RUN: not llvm-mc -triple riscv64 -mattr=+zhinx %s 2>&1 | FileCheck %s

# Invalid instructions
fmv.x.h t2, a2 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
fmv.h.x a5, t5 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.d.h a0, fa2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
fcvt.h.d a0, fa2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
