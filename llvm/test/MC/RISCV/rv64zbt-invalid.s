# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-b,experimental-zbt < %s 2>&1 | FileCheck %s

# Too few operands
fslw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
fsrw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
fsriw t0, t1, t2, 32 # CHECK: :[[@LINE]]:19: error: immediate must be an integer in the range [0, 31]
fsriw t0, t1, t2, -1 # CHECK: :[[@LINE]]:19: error: immediate must be an integer in the range [0, 31]
