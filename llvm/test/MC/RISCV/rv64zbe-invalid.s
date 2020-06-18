# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-b,experimental-zbe < %s 2>&1 | FileCheck %s

# Too few operands
bdepw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
bextw t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
