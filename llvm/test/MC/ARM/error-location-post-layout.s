@ RUN: not llvm-mc -triple armv7a--none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

@ Note: These errors are not always emitted in the order in which the relevant
@ source appears, this file is carefully ordered so that that is the case.

@ CHECK: <unknown>:0: error: expression could not be evaluated
  .set v1, -undef

  .comm common, 4
@ CHECK: <unknown>:0: error: Common symbol 'common' cannot be used in assignment expr
  .set v3, common

@ CHECK: <unknown>:0: error: symbol 'undef' could not be evaluated in a subtraction expression
  .set v2, a-undef
