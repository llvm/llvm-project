# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-b,experimental-zba < %s 2>&1 | FileCheck %s

# Too few operands
sh1add t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sh2add t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sh3add t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
