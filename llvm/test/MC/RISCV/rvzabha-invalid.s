# RUN: not llvm-mc -triple riscv32 -mattr=+a,+zabha < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+a,+zabha < %s 2>&1 | FileCheck %s

# Final operand must have parentheses
amoswap.b a1, a2, a3 # CHECK: :[[@LINE]]:19: error: expected '(' or optional integer offset
amomin.b a1, a2, 1 # CHECK: :[[@LINE]]:20: error: expected '(' after optional integer offset
amomin.b a1, a2, 1(a3) # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0

# Only .aq, .rl, and .aqrl suffixes are valid
amoxor.b.rlqa a2, a3, (a4) # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
amoor.b.aq.rl a4, a5, (a6) # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
amoor.b. a4, a5, (a6) # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic

# Non-zero offsets not supported for the third operand (rs1).
amocas.b a1, a3, 1(a5) # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0
amocas.h a0, a2, 2(a5) # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0
