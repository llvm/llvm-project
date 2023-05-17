# RUN: not llvm-mc --triple=loongarch64 --filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

  beq $a0, $a1, unaligned # CHECK: :[[#@LINE]]:3: error: fixup value must be 4-byte aligned
  beqz $a0, unaligned # CHECK: :[[#@LINE]]:3: error: fixup value must be 4-byte aligned
  b unaligned # CHECK: :[[#@LINE]]:3: error: fixup value must be 4-byte aligned
  .byte 0
unaligned:
  .byte 0
  .byte 0
  .byte 0

  beq $a0, $a1, out_of_range_b18 # CHECK: :[[#@LINE]]:3: error: fixup value out of range [-131072, 131071]
  .space 1<<18
out_of_range_b18:
  beqz $a0, out_of_range_b23 # CHECK: :[[#@LINE]]:3: error: fixup value out of range [-4194304, 4194303]
  .space 1<<23
out_of_range_b23:
  b out_of_range_b28 # CHECK: :[[#@LINE]]:3: error: fixup value out of range [-134217728, 134217727]
  .space 1<<28
out_of_range_b28:
