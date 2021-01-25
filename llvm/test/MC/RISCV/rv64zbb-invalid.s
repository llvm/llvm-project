# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-b,experimental-zbb < %s 2>&1 | FileCheck %s

# Too few operands
slow t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
srow t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sloiw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
sloiw t0, t1, 32 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
sloiw t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
# Too few operands
sroiw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
sroiw t0, t1, 32 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
sroiw t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
# Too many operands
clzw t0, t1, t2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
# Too many operands
ctzw t0, t1, t2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
# Too many operands
cpopw t0, t1, t2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
