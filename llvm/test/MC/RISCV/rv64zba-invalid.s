# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-b,experimental-zba < %s 2>&1 | FileCheck %s

# Too few operands
slli.uw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
slli.uw t0, t1, 64 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 63]
slli.uw t0, t1, -1 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 63]
# Too few operands
add.uw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sh1add.uw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sh2add.uw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sh3add.uw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
