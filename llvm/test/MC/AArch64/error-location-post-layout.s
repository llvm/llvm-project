// RUN: not llvm-mc -triple aarch64--none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

  .set v1, -undef
// CHECK: <unknown>:0: error: expression could not be evaluated

  .comm common, 4
  .set v3, common
// CHECK: 7:12: error: Common symbol 'common' cannot be used in assignment expr

  .set v2, a-undef
// CHECK: 10:13: error: symbol 'undef' could not be evaluated in a subtraction expression
