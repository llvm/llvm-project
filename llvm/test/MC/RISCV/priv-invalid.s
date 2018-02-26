# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s

mret 0x10 # CHECK: :[[@LINE]]:6: error: invalid operand for instruction

sfence.vma zero, a1, a2 # CHECK: :[[@LINE]]:22: error: invalid operand for instruction

sfence.vma a0, 0x10 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
