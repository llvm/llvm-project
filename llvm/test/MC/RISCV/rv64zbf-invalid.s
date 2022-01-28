# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-zbf < %s 2>&1 | FileCheck %s

# Too few operands
bfpw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
