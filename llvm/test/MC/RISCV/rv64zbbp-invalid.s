# RUN: not llvm-mc -triple riscv64 -mattr=+zbb,+experimental-zbp < %s 2>&1 | FileCheck %s

# Too few operands
rolw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
rorw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
roriw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
roriw t0, t1, 32 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
roriw t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
# Too few operands
packw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
packuw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
