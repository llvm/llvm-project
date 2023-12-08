// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - 2> %t.err > %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s

.quad (0x1234 + (4 * SOME_VALUE))
// CHECK-ERROR: error: expected relocatable expression
// CHECK-ERROR:               ^
