// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

cntb  w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cntb  w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntb  sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cntb  sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntb  z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cntb  z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Immediate not compatible with encode/decode function.

cntb  x0, all, mul #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: cntb  x0, all, mul #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntb  x0, all, mul #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: cntb  x0, all, mul #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntb  x0, all, mul #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: cntb  x0, all, mul #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate patterns

cntb  x0, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: cntb  x0, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntb  x0, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: cntb  x0, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntb  x0, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cntb  x0, vl512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntb  x0, #all, mul #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cntb  x0, #all, mul #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
