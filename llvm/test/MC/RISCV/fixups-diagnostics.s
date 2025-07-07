# RUN: not llvm-mc -triple riscv32 -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

  jal a0, far_distant # CHECK: :[[@LINE]]:11: error: fixup value out of range
  jal a0, unaligned # CHECK: :[[@LINE]]:11: error: fixup value must be 2-byte aligned

  blt t0, t1, unaligned # CHECK: :[[@LINE]]:15: error: fixup value must be 2-byte aligned

  .byte 0
unaligned:
  .byte 0
  .byte 0
  .byte 0

  .space 1<<20
far_distant:
