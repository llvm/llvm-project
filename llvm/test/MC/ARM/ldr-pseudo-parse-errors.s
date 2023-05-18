@RUN: not llvm-mc -triple=armv7-unknown-linux-gnueabi < %s  2>&1 | FileCheck %s
@RUN: not llvm-mc -triple=armv7-apple-darwin < %s  2>&1 | FileCheck %s

.text
bar:
  mov r0, =0x101
@ CHECK: error: unknown token in expression
@ CHECK: mov r0, =0x101
@ CHECK:         ^

