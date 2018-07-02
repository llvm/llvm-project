// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

uqdecb wsp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecb wsp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecb sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecb sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecb z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecb z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Operands not matching up (unsigned dec only has one register operand)

uqdecb x0, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecb x0, w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecb w0, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecb w0, w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecb x0, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecb x0, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Immediate not compatible with encode/decode function.

uqdecb x0, all, mul #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: uqdecb x0, all, mul #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecb x0, all, mul #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: uqdecb x0, all, mul #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecb x0, all, mul #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: uqdecb x0, all, mul #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate patterns

uqdecb x0, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecb x0, vl512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecb x0, vl9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecb x0, vl9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecb x0, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: uqdecb x0, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecb x0, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: uqdecb x0, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
