# RUN: not llvm-mc -triple riscv32 -mattr=+a < %s 2>&1 | FileCheck --match-full-lines %s
# RUN: not llvm-mc -triple riscv32 -mattr=+zalrsc < %s 2>&1 | FileCheck --match-full-lines %s

# Final operand must have parentheses
lr.w a4, a5 # CHECK: <stdin>:[[@LINE]]:10: error: expected '(' or optional integer offset

# lr only takes two operands
lr.w s0, (s1), s2 # CHECK: <stdin>:[[@LINE]]:16: error: unexpected extra operand for instruction
