# RUN: not llvm-mc -triple riscv32 -mattr=+xsmtvdot < %s 2>&1 \
# RUN:     | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xsmtvdot < %s 2>&1 \
# RUN:     | FileCheck %s

# NoSlide
smt.vmadot   v1, v2, v2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
smt.vmadotu  v1, v2, v2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
smt.vmadotsu v1, v2, v2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
smt.vmadotus v1, v2, v2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction

# Slide = 1
smt.vmadot1   v1, v2, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot1u  v1, v2, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot1su v1, v2, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot1us v1, v2, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot1   v2, v1, v2 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
smt.vmadot1u  v2, v1, v2 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
smt.vmadot1su v2, v1, v2 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
smt.vmadot1us v2, v1, v2 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
smt.vmadot1   v1, v3, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot1u  v1, v3, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot1su v1, v3, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot1us v1, v3, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction

# Slide = 2
smt.vmadot2   v1, v2, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot2u  v1, v2, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot2su v1, v2, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot2us v1, v2, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot2   v2, v1, v2 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
smt.vmadot2u  v2, v1, v2 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
smt.vmadot2su v2, v1, v2 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
smt.vmadot2us v2, v1, v2 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
smt.vmadot2   v1, v3, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot2u  v1, v3, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot2su v1, v3, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot2us v1, v3, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction

# Slide = 3
smt.vmadot3   v1, v2, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot3u  v1, v2, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot3su v1, v2, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot3us v1, v2, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot3   v2, v1, v2 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
smt.vmadot3u  v2, v1, v2 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
smt.vmadot3su v2, v1, v2 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
smt.vmadot3us v2, v1, v2 # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
smt.vmadot3   v1, v3, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot3u  v1, v3, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot3su v1, v3, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
smt.vmadot3us v1, v3, v2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
