# RUN: not llvm-mc -triple xtensa --mattr=+loop -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

  .align 4

  beq a0, a1, LBL1 # CHECK: :[[@LINE]]:15: error: fixup value out of range
LBL0:
  beqz a0, LBL2 # CHECK: :[[@LINE]]:12: error: fixup value out of range

  call0 LBL0 # CHECK: :[[@LINE]]:9: error: fixup value must be 4-byte aligned

  loop a3, LBL0 # CHECK: :[[@LINE]]:12: error: loop fixup value out of range

  .space 1<<8
LBL1:
  .space 1<<12
LBL2:
