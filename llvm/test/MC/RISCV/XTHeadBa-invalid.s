# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadba < %s 2>&1 | FileCheck --match-full-lines %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadba < %s 2>&1 | FileCheck --match-full-lines %s

# Too few operands
th.addsl t0, t1 # CHECK: <stdin>:[[@LINE]]:16: error: too few operands for instruction
# Too few operands
th.addsl t0, t1, t2 # CHECK: <stdin>:[[@LINE]]:20: error: too few operands for instruction
# Immediate operand out of range
th.addsl t0, t1, t2, 4 # CHECK: <stdin>:[[@LINE]]:22: error: immediate must be an integer in the range [0, 3]
# Immediate operand out of range
th.addsl t0, t1, t2, -1 # CHECK: <stdin>:[[@LINE]]:22: error: immediate must be an integer in the range [0, 3]
