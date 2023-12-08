# RUN: not llvm-mc -triple riscv64 -mattr=+zbs < %s 2>&1 | FileCheck %s

# Too few operands
bclr t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
bset t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
binv t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
bext t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
bclri t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
bclri t0, t1, 64 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 63]
bclri t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 63]
# Too few operands
bseti t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
bseti t0, t1, 64 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 63]
bseti t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 63]
# Too few operands
binvi t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
binvi t0, t1, 64 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 63]
binvi t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 63]
# Too few operands
bexti t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
bexti t0, t1, 64 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 63]
bexti t0, t1, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 63]
