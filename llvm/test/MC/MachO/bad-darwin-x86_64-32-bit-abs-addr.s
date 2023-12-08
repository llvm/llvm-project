// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - 2> %t.err > %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s

mov $_f, %rsi
// CHECK-ERROR: 32-bit absolute addressing is not supported in 64-bit mode
