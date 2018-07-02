// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

incw w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: incw w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

incw sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: incw sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// incw requires z0.s
incw z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: incw z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Immediate not compatible with encode/decode function.

incw x0, all, mul #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: incw x0, all, mul #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

incw x0, all, mul #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: incw x0, all, mul #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

incw x0, all, mul #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: incw x0, all, mul #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate patterns

incw x0, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: incw x0, vl512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

incw x0, vl9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: incw x0, vl9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

incw x0, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: incw x0, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

incw x0, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: incw x0, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
