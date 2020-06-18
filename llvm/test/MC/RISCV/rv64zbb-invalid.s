# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-b,experimental-zbb < %s 2>&1 | FileCheck %s

# Too few operands
addiwu t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
addiwu t0, t1, 2048 # CHECK: :[[@LINE]]:16: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]
addiwu t0, t1, -2049 # CHECK: :[[@LINE]]:16: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]
# Too few operands
slliu.w t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
slliu.w t0, t1, 64 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 63]
slliu.w t0, t1, -1 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 63]
# Too few operands
addwu t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
subwu t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
addu.w t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
subu.w t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
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
pcntw t0, t1, t2 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
