# RUN: not llvm-mc -triple riscv64 -mattr=+mem128ext %s 2>&1 | FileCheck %s

sq 1, a4, a0 # CHECK: :[[@LINE]]:4: error: invalid operand for instruction
sq a1, 4, a0 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
sq a1, a4, 0 # CHECK: :[[@LINE]]:12: error: invalid operand for instruction

lq 2, a0, a0 # CHECK: :[[@LINE]]:4: error: invalid operand for instruction
lq a2, 0, a0 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
lq a2, a0, 0 # CHECK: :[[@LINE]]:12: error: invalid operand for instruction

lq ra, 248(sp) # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
