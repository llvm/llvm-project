# RUN: not llvm-mc -triple riscv32 -mattr=+zimop < %s 2>&1 | FileCheck --match-full-lines %s

# Too few operands
mop.r.0 t0 # CHECK: <stdin>:[[@LINE]]:11: error: too few operands for instruction
# Too few operands
mop.rr.0 t0, t1 # CHECK: <stdin>:[[@LINE]]:16: error: too few operands for instruction
