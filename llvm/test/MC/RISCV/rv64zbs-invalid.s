# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-b,experimental-zbs < %s 2>&1 | FileCheck %s

# Too few operands
sbclrw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sbsetw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sbinvw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sbextw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sbclriw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
sbclriw t0, t1, 32 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
sbclriw t0, t1, -1 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
# Too few operands
sbsetiw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
sbsetiw t0, t1, 32 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
sbsetiw t0, t1, -1 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
# Too few operands
sbinviw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
sbinviw t0, t1, 32 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
sbinviw t0, t1, -1 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
