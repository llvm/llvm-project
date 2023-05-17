# RUN: not llvm-mc -triple xtensa -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

  .align 4

  beq a0, a1, LBL1 # CHECK: :[[@LINE]]:3: error: fixup value out of range
LBL0:
  beqz a0, LBL2 # CHECK: :[[@LINE]]:3: error: fixup value out of range

  call0 LBL0 # CHECK: :[[@LINE]]:3: error: fixup value must be 4-byte aligned

  .space 1<<8
LBL1:
  .space 1<<12
LBL2:
