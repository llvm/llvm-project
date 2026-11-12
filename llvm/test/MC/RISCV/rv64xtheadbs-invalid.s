# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadbs < %s 2>&1 | FileCheck --match-full-lines %s

# Immediate operand out of range
th.tst t0, t1, 64 # CHECK: <stdin>:[[@LINE]]:16: error: immediate must be an integer in the range [0, 63]
th.tst t0, t1, -1 # CHECK: <stdin>:[[@LINE]]:16: error: immediate must be an integer in the range [0, 63]
