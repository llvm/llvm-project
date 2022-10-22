# RUN: not llvm-mc --triple=loongarch64 --filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

  beq $a0, $a1, unaligned # CHECK: :[[#@LINE]]:3: error: fixup value must be 4-byte aligned
  beqz $a0, unaligned # CHECK: :[[#@LINE]]:3: error: fixup value must be 4-byte aligned
  b unaligned # CHECK: :[[#@LINE]]:3: error: fixup value must be 4-byte aligned
  .byte 0
unaligned:
  .byte 0
  .byte 0
  .byte 0

  beq $a0, $a1, out_of_range_b16 # CHECK: :[[#@LINE]]:3: error: fixup value out of range
  .space 1<<18
out_of_range_b16:
  beqz $a0, out_of_range_b21 # CHECK: :[[#@LINE]]:3: error: fixup value out of range
  .space 1<<23
out_of_range_b21:
  b out_of_range_b26 # CHECK: :[[#@LINE]]:3: error: fixup value out of range
  .space 1<<28
out_of_range_b26:
