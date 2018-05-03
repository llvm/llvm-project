// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of lower bound [-8, 7].

stnt1b z23.b, p0, [x13, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: stnt1b z23.b, p0, [x13, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1b z29.b, p0, [x3, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: stnt1b z29.b, p0, [x3, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid source type.

stnt1b z0.h, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stnt1b z0.h, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1b z0.s, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stnt1b z0.s, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1b z0.d, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stnt1b z0.d, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// invalid predicate

stnt1b z27.b, p8, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: stnt1b z27.b, p8, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1b z0.h, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stnt1b z0.h, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

stnt1b { }, p0, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: stnt1b { }, p0, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1b { z1.b, z2.b }, p0, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: stnt1b { z1.b, z2.b }, p0, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1b { v0.2d }, p0, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: stnt1b { v0.2d }, p0, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
