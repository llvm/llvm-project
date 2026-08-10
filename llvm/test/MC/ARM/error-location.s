@ RUN: not llvm-mc -triple armv7a--none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple thumbv7a--none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

@ Note: These errors are not always emitted in the order in which the relevant
@ source appears, this file is carefully ordered so that that is the case.

  .text
@ CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: symbol 'undef' can not be undefined in a subtraction expression
  .word (0-undef)

@ CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  .word -undef

@ CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: symbol 'undef' can not be undefined in a subtraction expression
  adr r0, #a-undef

@ CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Cannot represent a difference across sections
  .word x_a - y_a

@ CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid fixup for {{ARM|Thumb}} MOVT instruction
  movt r9, :upper16: bar(PREL31)

@ CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid fixup for {{ARM|Thumb}} MOVW instruction
  movw r9, :lower16: bar(PREL31)

@ CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid fixup for 1-byte data relocation
  .byte f30(PLT)

@ CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid fixup for 2-byte data relocation
  .hword f30(PLT)

@ CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid fixup for 4-byte data relocation
  .word f30(PLT)

@ CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid fixup for 4-byte pc-relative data relocation
  .word x_a(PLT) - label1
label1:

w:
  .word 0
  .weak w


  .section sec_x
x_a:
  .word 0


  .section sec_y
y_a:
  .word 0
