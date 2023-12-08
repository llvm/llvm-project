# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadsync < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadsync < %s 2>&1 | FileCheck %s

th.sfence.vmas t0 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.sync t0 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
th.sync.s t0 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
th.sync.i t0 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
th.sync.is t0 # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
