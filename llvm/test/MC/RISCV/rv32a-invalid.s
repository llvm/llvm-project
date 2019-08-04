# RUN: not llvm-mc -triple riscv32 -mattr=+a < %s 2>&1 | FileCheck %s

# Final operand must have parentheses
amoswap.w a1, a2, a3 # CHECK: :[[@LINE]]:19: error: expected '(' or optional integer offset
amomin.w a1, a2, 1 # CHECK: :[[@LINE]]:20: error: expected '(' after optional integer offset
amomin.w a1, a2, 1(a3) # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0
lr.w a4, a5 # CHECK: :[[@LINE]]:10: error: expected '(' or optional integer offset

# Only .aq, .rl, and .aqrl suffixes are valid
amoxor.w.rlqa a2, a3, (a4) # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
amoor.w.aq.rl a4, a5, (a6) # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
amoor.w. a4, a5, (a6) # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic

# lr only takes two operands
lr.w s0, (s1), s2 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction

# Note: errors for use of RV64A instructions for RV32 are checked in
# rv64a-valid.s
