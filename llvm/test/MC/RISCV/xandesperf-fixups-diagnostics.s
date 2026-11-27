# RUN: llvm-mc --verify -triple riscv32 -filetype obj -mattr=+xandesperf -o /dev/null < %s

  nds.bbc t0, 7, far_distant # expected-error:18 {{fixup value out of range}}
  nds.bbc t0, 7, unaligned # expected-error:18 {{fixup value must be 2-byte aligned}}

  .byte 0
unaligned:
  .byte 0
  .byte 0
  .byte 0

  .space 1<<10
far_distant:
