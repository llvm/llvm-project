# RUN: not llvm-mc -triple riscv32 -mattr=+zbc < %s 2>&1 | FileCheck --match-full-lines %s

# Too few operands
clmul t0, t1 # CHECK: <stdin>:[[@LINE]]:13: error: too few operands for instruction
# Too few operands
clmulr t0, t1 # CHECK: <stdin>:[[@LINE]]:14: error: too few operands for instruction
# Too few operands
clmulh t0, t1 # CHECK: <stdin>:[[@LINE]]:14: error: too few operands for instruction
