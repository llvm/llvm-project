// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

uqincp sp, p0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqincp sp, p0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincp z0.b, p0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: uqincp z0.b, p0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincp x0, p0.b, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqincp x0, p0.b, w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincp x0, p0.b, x1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqincp x0, p0.b, x1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate operand

uqincp x0, p0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: uqincp x0, p0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincp x0, p0/z
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: uqincp x0, p0/z
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincp x0, p0/m
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: uqincp x0, p0/m
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqincp x0, p0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: uqincp x0, p0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
