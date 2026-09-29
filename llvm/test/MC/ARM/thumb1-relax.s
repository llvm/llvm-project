@ RUN: not llvm-mc -triple thumbv6m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple thumbv6m-none-eabi -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s

Lhere:
@ CHECK: out of range pc-relative fixup value
  ldr r0, Lhere

@ CHECK: out of range pc-relative fixup value
  b Lfar2

@ CHECK: out of range pc-relative fixup value
  bne Lfar1

@ CHECK: out of range pc-relative fixup value
  ldr r0, Lfar2

@ CHECK: misaligned pc-relative fixup value
  adr r0, Lmisaligned

@ CHECK: misaligned pc-relative fixup value
  ldr r0, Lmisaligned

  .balign 4
  .short 0
Lmisaligned:
  .word 42

  .space 256
Lfar1:
  .word 42

  .space 2050
Lfar2:
  .word 42

