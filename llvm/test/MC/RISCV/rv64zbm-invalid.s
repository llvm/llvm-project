# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-zbm < %s 2>&1 | FileCheck %s

# Too many operands
bmatflip t0, t1, t2 # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
# Too few operands
bmator t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
bmatxor t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
