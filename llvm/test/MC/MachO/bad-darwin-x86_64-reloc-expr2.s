// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - 2> %t.err > %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s

_Z:
.long (_a+4)-_Z
// CHECK-ERROR: error: unsupported relocation with subtraction expression, symbol '_a' can not be undefined in a subtraction expression
