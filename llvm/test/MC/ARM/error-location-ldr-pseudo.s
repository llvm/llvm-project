@ RUN: not llvm-mc -triple armv7a--none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

  .text
@ CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  ldr r0, =(-undef)
