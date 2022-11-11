# RUN: not llvm-mc -triple riscv64 -mattr=+zbb < %s 2>&1 | FileCheck %s

# Too many operands
clzw t0, t1, t2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
# Too many operands
ctzw t0, t1, t2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
# Too many operands
cpopw t0, t1, t2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
# Too few operands
rolw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
rorw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
roriw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
roriw t0, t1, 32 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
roriw t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
rori t0, t1, 64 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 63]
rori t0, t1, -1 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 63]
