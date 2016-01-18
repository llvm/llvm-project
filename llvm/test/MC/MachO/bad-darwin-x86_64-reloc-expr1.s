// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - 2> %t.err > %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s

_Z:
.long (_Z+4)-_b
// CHECK-ERROR: error: unsupported relocation with subtraction expression, symbol '_b' can not be undefined in a subtraction expression
