# RUN: not llvm-mc -triple riscv64 -mattr=+a < %s 2>&1 | FileCheck %s

# Final operand must have parentheses
lr.d a4, a5 # CHECK: :[[@LINE]]:10: error: expected '(' or optional integer offset

# lr only takes two operands
lr.d s0, (s1), s2 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
