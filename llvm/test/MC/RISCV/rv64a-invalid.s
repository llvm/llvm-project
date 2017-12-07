# RUN: not llvm-mc -triple riscv64 -mattr=+a < %s 2>&1 | FileCheck %s

# Final operand must have parentheses
amoswap.d a1, a2, a3 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
amomin.d a1, a2, 1 # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
lr.d a4, a5 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

# Only .aq, .rl, and .aqrl suffixes are valid
amoxor.d.rlqa a2, a3, (a4) # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
amoor.d.aq.rl a4, a5, (a6) # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
amoor.d. a4, a5, (a6) # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic

# lr only takes two operands
lr.d s0, (s1), s2 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
