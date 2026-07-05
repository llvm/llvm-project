# RUN: not llvm-mc -triple riscv32 -mattr=+d -mattr=+xtheadfmemidx < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+d -mattr=+xtheadfmemidx < %s 2>&1 | FileCheck %s

th.flrd fa0, a1, a2, 5     # CHECK: :[[@LINE]]:22: error: immediate must be an integer in the range [0, 3]
th.flrd a0, a1, a2, 3      # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
th.flrw 0(fa0), a1, a2, 0  # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
th.flrw fa0, 4(a1), a2, 3  # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
th.fsrd fa0, a1, -1(a2), 0 # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
th.fsrd fa0, a1, a2, -3    # CHECK: :[[@LINE]]:22: error: immediate must be an integer in the range [0, 3]
