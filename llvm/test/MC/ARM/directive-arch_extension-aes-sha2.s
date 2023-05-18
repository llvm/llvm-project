@ RUN: not llvm-mc -triple armv8-eabi -filetype asm < %s 2> %t  | FileCheck %s
@ RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s
@ RUN: not llvm-mc -triple thumbv8-eabi -filetype asm < %s 2> %t | FileCheck %s
@ RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s

  .syntax unified

  .arch_extension aes
  .arch_extension sha2

  .type crypto,%function
crypto:
  aesd.8 q0, q1
  sha1c.32 q0, q1, q2

@CHECK-LABEL: crypto:
@CHECK:	aesd.8 q0, q1
@CHECK:	sha1c.32 q0, q1, q2

  .arch_extension noaes
  .arch_extension nosha2

  .type nocrypto,%function
nocrypto:
  aesd.8 q0, q1
  sha1c.32 q0, q1, q2

@CHECK-ERROR: error: instruction requires: aes
@CHECK-ERROR: aesd.8 q0, q1
@CHECK-ERROR: ^

@CHECK-ERROR: error: instruction requires: sha2
@CHECK-ERROR: sha1c.32 q0, q1, q2
@CHECK-ERROR: ^
