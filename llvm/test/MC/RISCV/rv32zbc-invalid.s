# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-b,experimental-zbc < %s 2>&1 | FileCheck %s

# Too few operands
clmul t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
clmulr t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
clmulh t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
