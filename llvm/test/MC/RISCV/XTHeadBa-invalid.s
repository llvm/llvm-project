# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadba < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadba < %s 2>&1 | FileCheck %s

# Too few operands
th.addsl t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
th.addsl t0, t1, t2 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
th.addsl t0, t1, t2, 4 # CHECK: :[[@LINE]]:22: error: immediate must be an integer in the range [0, 3]
# Immediate operand out of range
th.addsl t0, t1, t2, -1 # CHECK: :[[@LINE]]:22: error: immediate must be an integer in the range [0, 3]
