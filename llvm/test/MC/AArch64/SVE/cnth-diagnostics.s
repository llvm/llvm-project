// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

cnth  w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cnth  w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cnth  sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cnth  sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cnth  z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cnth  z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Immediate not compatible with encode/decode function.

cnth  x0, all, mul #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: cnth  x0, all, mul #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cnth  x0, all, mul #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: cnth  x0, all, mul #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cnth  x0, all, mul #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: cnth  x0, all, mul #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate patterns

cnth  x0, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: cnth  x0, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cnth  x0, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: cnth  x0, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cnth  x0, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cnth  x0, vl512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
