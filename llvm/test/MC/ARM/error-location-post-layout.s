@ RUN: not llvm-mc -triple armv7a--none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

  .set v1, -undef
@ CHECK: 3:12: error: expression could not be evaluated

  .comm common, 4
  .set v3, common
@ CHECK: 7:12: error: Common symbol 'common' cannot be used in assignment expr

  .set v2, a-undef
@ CHECK-DAG: 10:13: error: symbol 'undef' could not be evaluated in a subtraction expression

  .equ STACK_START, (a + undef)
@ CHECK-DAG: 13:24: error: expression could not be evaluated
