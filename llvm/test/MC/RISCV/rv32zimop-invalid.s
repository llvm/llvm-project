# RUN: not llvm-mc -triple riscv32 -mattr=+zimop < %s 2>&1 | FileCheck %s

# Too few operands
mop.r.0 t0 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
mop.rr.0 t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
