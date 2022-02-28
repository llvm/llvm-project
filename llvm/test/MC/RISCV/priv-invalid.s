# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s

mret 0x10 # CHECK: :[[@LINE]]:6: error: invalid operand for instruction

sfence.vma zero, a1, a2 # CHECK: :[[@LINE]]:22: error: invalid operand for instruction

sfence.vma a0, 0x10 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction

sinval.vma zero, a1, a2 # CHECK: :[[@LINE]]:22: error: invalid operand for instruction

sinval.vma a0, 0x10 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction

sfence.w.inval 0x10 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction

sfence.inval.ir 0x10 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction

hfence.vvma zero, a1, a2 # CHECK: :[[@LINE]]:23: error: invalid operand for instruction

hfence.vvma a0, 0x10 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction

hfence.gvma zero, a1, a2 # CHECK: :[[@LINE]]:23: error: invalid operand for instruction

hfence.gvma a0, 0x10 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction

hinval.vvma zero, a1, a2 # CHECK: :[[@LINE]]:23: error: invalid operand for instruction

hinval.vvma a0, 0x10 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction

hinval.gvma zero, a1, a2 # CHECK: :[[@LINE]]:23: error: invalid operand for instruction

hinval.gvma a0, 0x10 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
