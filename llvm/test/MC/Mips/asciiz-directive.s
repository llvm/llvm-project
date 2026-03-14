# RUN: llvm-mc -triple mips-unknown-linux %s | FileCheck %s
# .asciiz is exactly the same as .asciz, except it's MIPS-specific.

t1:
  .asciiz
# CHECK-LABEL: t1

t2:
  .asciiz "a"
# CHECK-LABEL: t2
# CHECK: .byte 97
# CHECK: .byte 0

t3:
  .asciiz "a", "b", "c"
# CHECK-LABEL: t3
# CHECK: .byte 97
# CHECK: .byte 0
# CHECK: .byte 98
# CHECK: .byte 0
# CHECK: .byte 99
# CHECK: .byte 0

t4:
  .asciiz "abcdefghijklmnop"
# CHECK-LABEL: t4
# CHECK: .ascii "abcdefghijklmnop"
# CHECK: .byte 0
