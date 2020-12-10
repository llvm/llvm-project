# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-zfh < %s 2>&1 | \
# RUN:   FileCheck %s

# Integer registers where FP regs are expected
fcvt.l.h ft0, a0 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
fcvt.lu.h ft1, a1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction

# FP registers where integer regs are expected
fcvt.h.l a2, ft2 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
fcvt.h.lu a3, ft3 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
