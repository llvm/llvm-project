// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - 2> %t.err > %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s

.quad _foo - _bar
// CHECK-ERROR: error: unsupported relocation with subtraction expression

_Y:
.long (_Y+4)-_b
// CHECK-ERROR: error: unsupported relocation with subtraction expression, symbol '_b' can not be undefined in a subtraction expression

_Z:
.long (_a+4)-_Z
// CHECK-ERROR: error: unsupported relocation with subtraction expression, symbol '_a' can not be undefined in a subtraction expression
