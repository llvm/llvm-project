# RUN: not llvm-mc -triple riscv32 -mattr=+zbkx < %s 2>&1 | FileCheck --match-full-lines %s

# Too few operands
xperm8 t0, t1 # CHECK: <stdin>:[[@LINE]]:14: error: too few operands for instruction
# Too few operands
xperm4 t0, t1 # CHECK: <stdin>:[[@LINE]]:14: error: too few operands for instruction
