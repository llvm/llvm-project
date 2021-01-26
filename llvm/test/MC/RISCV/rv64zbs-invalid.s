# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-b,experimental-zbs < %s 2>&1 | FileCheck %s

# Too few operands
bclrw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
bsetw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
binvw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
bextw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
bclriw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
bclriw t0, t1, 32 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
bclriw t0, t1, -1 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
# Too few operands
bsetiw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
bsetiw t0, t1, 32 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
bsetiw t0, t1, -1 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
# Too few operands
binviw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
binviw t0, t1, 32 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
binviw t0, t1, -1 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
