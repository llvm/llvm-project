# RUN: not llvm-mc --triple=loongarch64 --filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

  beq $a0, $a1, far_distant # CHECK: :[[#@LINE]]:3: error: fixup value out of range
  bne $a0, $a1, unaligned # CHECK: :[[#@LINE]]:3: error: fixup value must be 4-byte aligned

  bnez $a0, unaligned # CHECK: :[[#@LINE]]:3: error: fixup value must be 4-byte aligned
  beqz $a0, far_distant_bz # CHECK: :[[#@LINE]]:3: error: fixup value out of range

  b unaligned # CHECK: :[[#@LINE]]:3: error: fixup value must be 4-byte aligned

  .byte 0
unaligned:
  .byte 0
  .byte 0
  .byte 0

  .space 1<<16
distant:
  .space 1<<18
far_distant:

  .byte 0
unaligned_bz:
  .byte 0
  .byte 0
  .byte 0
distant_bz:
  .space 1<<23
far_distant_bz:
