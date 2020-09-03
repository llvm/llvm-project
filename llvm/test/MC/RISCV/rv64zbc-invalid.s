# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-b,experimental-zbc < %s 2>&1 | FileCheck %s

# Too few operands
clmulw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
clmulrw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
clmulhw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
