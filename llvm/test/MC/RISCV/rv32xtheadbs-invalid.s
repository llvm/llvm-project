# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadbs < %s 2>&1 | FileCheck --match-full-lines %s

# Too few operands
th.tst t0, t1 # CHECK: <stdin>:[[@LINE]]:14: error: too few operands for instruction
# Immediate operand out of range
th.tst t0, t1, 32 # CHECK: <stdin>:[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
th.tst t0, t1, -1 # CHECK: <stdin>:[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
