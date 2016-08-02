// RUN: not llvm-mc -triple aarch64--none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

// Note: These errors are not always emitted in the order in which the relevant
// source appears, this file is carefully ordered so that that is the case.

  .text
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: symbol 'undef' can not be undefined in a subtraction expression
  .word (0-undef)

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  .word -undef

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: No relocation available to represent this relative expression
  adr x0, #a-undef

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Cannot represent a difference across sections
  .word x_a - y_a

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Cannot represent a subtraction with a weak symbol
  .word a - w

// CHECK: <unknown>:0: error: expression could not be evaluated
  .set v1, -undef

  .comm common, 4
// CHECK: <unknown>:0: error: Common symbol 'common' cannot be used in assignment expr
  .set v3, common

// CHECK: <unknown>:0: error: Undefined temporary symbol
  .word 5f

// CHECK: <unknown>:0: error: symbol 'undef' could not be evaluated in a subtraction expression
  .set v2, a-undef



w:
  .word 0
  .weak w


  .section sec_x
x_a:
  .word 0


  .section sec_y
y_a:
  .word 0
