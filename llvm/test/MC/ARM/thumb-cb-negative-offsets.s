@ RUN: not llvm-mc -triple thumbv7m-none-eabi      -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple thumbv8m.base-none-eabi -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s

label0:
  .word 4

@ CHECK: out of range pc-relative fixup value
  cbz r0, label0
@ CHECK: out of range pc-relative fixup value
  cbnz r0, label0

@ CHECK: out of range pc-relative fixup value
  cbz r0, label1
@ CHECK: out of range pc-relative fixup value
  cbnz r0, label1

  .space 1000
label1:
  nop

@ CHECK: out of range pc-relative fixup value
  cbz r0, label2
  .space 130
label2:
  nop

@ CHECK-NOT: label3
  cbnz r0, label3
  .space 128
label3:
  nop
