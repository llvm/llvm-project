// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

sqdecw w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdecw w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecw wsp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdecw wsp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecw sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdecw sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecw z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqdecw z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Operands not matching up

sqdecw x0, w1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be 32-bit form of destination register
// CHECK-NEXT: sqdecw x0, w1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecw x0, x1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdecw x0, x1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Immediate not compatible with encode/decode function.

sqdecw x0, all, mul #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqdecw x0, all, mul #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecw x0, all, mul #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqdecw x0, all, mul #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecw x0, all, mul #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqdecw x0, all, mul #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate patterns

sqdecw x0, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdecw x0, vl512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecw x0, vl9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdecw x0, vl9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecw x0, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: sqdecw x0, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdecw x0, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: sqdecw x0, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
