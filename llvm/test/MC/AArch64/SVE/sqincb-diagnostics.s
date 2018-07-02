// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

sqincb w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincb w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincb wsp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincb wsp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincb sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincb sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincb z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincb z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Operands not matching up

sqincb x0, w1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be 32-bit form of destination register
// CHECK-NEXT: sqincb x0, w1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincb x0, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincb x0, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Immediate not compatible with encode/decode function.

sqincb x0, all, mul #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqincb x0, all, mul #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincb x0, all, mul #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqincb x0, all, mul #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincb x0, all, mul #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqincb x0, all, mul #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate patterns

sqincb x0, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincb x0, vl512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincb x0, vl9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincb x0, vl9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincb x0, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: sqincb x0, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincb x0, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: sqincb x0, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
