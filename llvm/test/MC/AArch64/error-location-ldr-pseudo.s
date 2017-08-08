// RUN: not llvm-mc -triple aarch64--none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

  .text
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  ldr x0, =(-undef)
