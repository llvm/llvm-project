@ RUN: llvm-mc -triple thumbv7-apple-macho -filetype=obj -o %t %s
@ RUN: llvm-objdump -d --triple=thumbv7 %t | FileCheck %s

@ CHECK: cbnz r0, 0x4 <label4> @ imm = #0
  .thumb_func label4
  cbnz r0, label4
  .space 2
label4:
