# RUN: not llvm-mc -triple riscv32 -mattr=+zbkc < %s 2>&1 | FileCheck --match-full-lines %s

# Too few operands
clmul t0, t1 # CHECK: <stdin>:[[@LINE]]:13: error: too few operands for instruction
# Too few operands
clmulh t0, t1 # CHECK: <stdin>:[[@LINE]]:14: error: too few operands for instruction

# Undefined zbc instruction in zbkc
clmulr t0, t1, t2 # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'Zbc' (Carry-Less Multiplication)
