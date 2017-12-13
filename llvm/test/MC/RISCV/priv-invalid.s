# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s

mret 0x10 # CHECK: :[[@LINE]]:6: error: invalid operand for instruction

sfence.vma zero # CHECK: :[[@LINE]]:1: error: too few operands for instruction

sfence.vma a0, 0x10 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
