// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Only .b is supported

brkas  p0.s, p15/z, p15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: brkas  p0.s, p15/z, p15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// flag-setting variant does not have merging predication

brkas  p0.b, p15/m, p15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: brkas  p0.b, p15/m, p15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
