// RUN: llvm-mc -triple x86_64-linux-gnu %s | FileCheck %s
.macro abc arg:vararg=nop
  \arg
.endm

.macro abcd arg0=%eax arg1:vararg=%ebx
  movl \arg0, \arg1
.endm

.text

// CHECK: nop
  abc
// CHECK: movl %eax, %ebx
  abcd ,
