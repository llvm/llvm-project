# RUN: not llvm-mc -triple riscv32 -mattr=+zbkc < %s 2>&1 | FileCheck %s

# Too few operands
clmul t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
clmulh t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction

# Undefined zbc instruction in zbkc
clmulr t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zbc' (Carry-Less Multiplication)
