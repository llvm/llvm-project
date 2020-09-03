# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-b,experimental-zbp < %s 2>&1 | FileCheck %s

# Too few operands
gorc t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
grev t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
gorci t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
gorci t0, t1, 32 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
gorci t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
# Too few operands
grevi t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
grevi t0, t1, 32 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
grevi t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
# Too few operands
shfl t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
unshfl t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
shfli t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
shfli t0, t1, 16 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 15]
shfli t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 15]
# Too few operands
unshfli t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
unshfli t0, t1, 16 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 15]
unshfli t0, t1, -1 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 15]
