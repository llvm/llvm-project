# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadmemidx < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadmemidx < %s 2>&1 | FileCheck %s

th.ldia		0(a0), (a1), 0, 0  # CHECK: :[[@LINE]]:23: error: invalid operand for instruction
th.ldib		a0, 2(a1), 15, 1   # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
th.lwia		a0, (a1), 30, 2    # CHECK: :[[@LINE]]:20: error: immediate must be an integer in the range [-16, 15]
th.lwib		a0, (a1), -16, 43  # CHECK: :[[@LINE]]:25: error: immediate must be an integer in the range [0, 3]
th.lhib		a0, (a1), -17, 3   # CHECK: :[[@LINE]]:20: error: immediate must be an integer in the range [-16, 15]
th.lrb		-2(a0), a1, a2, 0  # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
th.lrw		a0, 3(a1), a2, 1   # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
th.lrw		a0, a1, 4(a2), 2   # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
th.lrh		a0, a1, a2, 5      # CHECK: :[[@LINE]]:21: error: immediate must be an integer in the range [0, 3]
th.lrhu		a0, a1, a2, -1     # CHECK: :[[@LINE]]:22: error: immediate must be an integer in the range [0, 3]
