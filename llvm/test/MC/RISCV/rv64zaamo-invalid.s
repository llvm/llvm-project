# RUN: not llvm-mc -triple riscv64 -mattr=+a < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-zaamo < %s 2>&1 | FileCheck %s

# Final operand must have parentheses
amoswap.d a1, a2, a3 # CHECK: :[[@LINE]]:19: error: expected '(' or optional integer offset
amomin.d a1, a2, 1 # CHECK: :[[@LINE]]:20: error: expected '(' after optional integer offset
amomin.d a1, a2, 1(a3) # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0

# Only .aq, .rl, and .aqrl suffixes are valid
amoxor.d.rlqa a2, a3, (a4) # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
amoor.d.aq.rl a4, a5, (a6) # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
amoor.d. a4, a5, (a6) # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
