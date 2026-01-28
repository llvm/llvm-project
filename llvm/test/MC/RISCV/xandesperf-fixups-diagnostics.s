# RUN: not llvm-mc -triple riscv32 -filetype obj -mattr=+xandesperf < %s -o /dev/null 2>&1 | FileCheck %s

  nds.bbc t0, 7, far_distant # CHECK: :[[@LINE]]:18: error: fixup value out of range
  nds.bbc t0, 7, unaligned # CHECK: :[[@LINE]]:18: error: fixup value must be 2-byte aligned

  .byte 0
unaligned:
  .byte 0
  .byte 0
  .byte 0

  .space 1<<10
far_distant:
