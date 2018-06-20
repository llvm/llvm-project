// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

uqdech w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdech w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdech wsp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdech wsp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdech sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdech sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Immediate not compatible with encode/decode function.

uqdech x0, all, mul #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: uqdech x0, all, mul #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdech x0, all, mul #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: uqdech x0, all, mul #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdech x0, all, mul #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: uqdech x0, all, mul #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate patterns

uqdech x0, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: uqdech x0, vl512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdech x0, vl9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: uqdech x0, vl9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdech x0, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: uqdech x0, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdech x0, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: uqdech x0, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
