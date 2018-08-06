// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

sqdecp sp, p0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdecp sp, p0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecp z0.b, p0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqdecp z0.b, p0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecp w0, p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdecp w0, p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecp x0, p0.b, x1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdecp x0, p0.b, x1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecp x0, p0.b, w1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be 32-bit form of destination register
// CHECK-NEXT: sqdecp x0, p0.b, w1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate operand

sqdecp x0, p0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: sqdecp x0, p0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecp x0, p0/z
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: sqdecp x0, p0/z
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecp x0, p0/m
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: sqdecp x0, p0/m
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecp x0, p0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: sqdecp x0, p0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
