@ RUN: not llvm-mc -n -triple armv7-apple-darwin10 %s -filetype=obj -o - 2> %t.err > %t
@ RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s
@ rdar://15586725
.text
    ldr r3, L___fcommon
.section myseg, mysect
L___fcommon: 
    .word 0

c:
  .word a - b
@ CHECK-ERROR: symbol 'a' can not be undefined in a subtraction expression
  .word c - b
@ CHECK-ERROR: symbol 'b' can not be undefined in a subtraction expression
