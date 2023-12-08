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

hlv.b a0, 0x10 # CHECK: :[[@LINE]]:16: error: expected '(' after optional integer offset

hlv.b a0, a1 # CHECK: :[[@LINE]]:11: error: expected '(' or optional integer offset

hlv.b a0, 1(a1) # CHECK: :[[@LINE]]:11: error: optional integer offset must be 0

hlv.bu a0, 0x10 # CHECK: :[[@LINE]]:17: error: expected '(' after optional integer offset

hlv.bu a0, a1 # CHECK: :[[@LINE]]:12: error: expected '(' or optional integer offset

hlv.bu a0, 1(a1) # CHECK: :[[@LINE]]:12: error: optional integer offset must be 0

hlv.h a0, 0x10 # CHECK: :[[@LINE]]:16: error: expected '(' after optional integer offset

hlv.h a0, a1 # CHECK: :[[@LINE]]:11: error: expected '(' or optional integer offset

hlv.h a0, 1(a1) # CHECK: :[[@LINE]]:11: error: optional integer offset must be 0

hlv.hu a0, 0x10 # CHECK: :[[@LINE]]:17: error: expected '(' after optional integer offset

hlv.hu a0, a1 # CHECK: :[[@LINE]]:12: error: expected '(' or optional integer offset

hlv.hu a0, 1(a1) # CHECK: :[[@LINE]]:12: error: optional integer offset must be 0

hlvx.hu a0, 0x10 # CHECK: :[[@LINE]]:18: error: expected '(' after optional integer offset
    
hlvx.hu a0, a1 # CHECK: :[[@LINE]]:13: error: expected '(' or optional integer offset
    
hlvx.hu a0, 1(a1) # CHECK: :[[@LINE]]:13: error: optional integer offset must be 0

hlv.w a0, 0x10 # CHECK: :[[@LINE]]:16: error: expected '(' after optional integer offset

hlv.w a0, a1 # CHECK: :[[@LINE]]:11: error: expected '(' or optional integer offset

hlv.w a0, 1(a1) # CHECK: :[[@LINE]]:11: error: optional integer offset must be 0

hlvx.wu a0, 0x10 # CHECK: :[[@LINE]]:18: error: expected '(' after optional integer offset

hlvx.wu a0, a1 # CHECK: :[[@LINE]]:13: error: expected '(' or optional integer offset

hlvx.wu a0, 1(a1) # CHECK: :[[@LINE]]:13: error: optional integer offset must be 0

hsv.b a0, 0x10 # CHECK: :[[@LINE]]:16: error: expected '(' after optional integer offset

hsv.b a0, a1 # CHECK: :[[@LINE]]:11: error: expected '(' or optional integer offset

hsv.b a0, 1(a1) # CHECK: :[[@LINE]]:11: error: optional integer offset must be 0

hsv.h a0, 0x10 # CHECK: :[[@LINE]]:16: error: expected '(' after optional integer offset

hsv.h a0, a1 # CHECK: :[[@LINE]]:11: error: expected '(' or optional integer offset

hsv.h a0, 1(a1) # CHECK: :[[@LINE]]:11: error: optional integer offset must be 0

hsv.w a0, 0x10 # CHECK: :[[@LINE]]:16: error: expected '(' after optional integer offset

hsv.w a0, a1 # CHECK: :[[@LINE]]:11: error: expected '(' or optional integer offset

hsv.w a0, 1(a1) # CHECK: :[[@LINE]]:11: error: optional integer offset must be 0
