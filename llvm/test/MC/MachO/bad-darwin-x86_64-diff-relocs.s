// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - 2> %t.err > %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s

.quad _foo - _bar
// CHECK-ERROR: unsupported relocation with subtraction expression
