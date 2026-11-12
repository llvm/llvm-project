@ RUN: llvm-mc < %s -triple armv7-unknown-unknown -filetype=obj | llvm-objdump -d - | FileCheck %s

@ Test that makes sure both label and immediate expression
@ are evaluated to the same values.

AES_Te:
  .word 1,2,3,4,5,6
  .word 1,2,3,4,5,6
  .word 1,2,3,4,5,6
  .word 1,2,3,4,5,6
  .word 1,2,3,4,5,6
  .word 1,2,3,4,5,6
  .word 1,2,3,4,5,6
  .word 1,2,3,4,5,6
  .word 1,2,3,4,5,6
  .word 1,2,3,4,5,6
  .word 1,2,3,4,5,6
@ CHECK: <AES_encrypt>:
AES_encrypt:
@ CHECK: sub	r10, r3, #264
  sub r10,r3,#(AES_encrypt-AES_Te)
@ CHECK: sub	r10, r3, #264
  sub r10,r3,#(6*11*4)
Data:
@ CHECK: 08 01 00 00
  .word (AES_encrypt-AES_Te)
@ CHECK: 08 01 00 00
  .word (6*11*4)

