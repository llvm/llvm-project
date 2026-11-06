# RUN: llvm-mc --verify -triple riscv32 -filetype obj -o /dev/null < %s

  jal a0, far_distant # expected-error:11 {{fixup value out of range}}
  jal a0, unaligned # expected-error:11 {{fixup value must be 2-byte aligned}}

  blt t0, t1, unaligned # expected-error:15 {{fixup value must be 2-byte aligned}}

  .byte 0
unaligned:
  .byte 0
  .byte 0
  .byte 0

  .space 1<<20
far_distant:
