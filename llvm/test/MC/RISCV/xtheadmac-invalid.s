# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadmac < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadmac < %s 2>&1 | FileCheck %s

th.mula  t0, t1, 0(t2)  # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
th.muls  t0, 12, t2     # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
th.mulsh t0, t1, 34     # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
th.mulsw t0, -56, t2    # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
