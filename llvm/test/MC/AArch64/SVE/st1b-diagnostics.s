// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of upper bound [-8, 7].

st1b z10.b, p4, [x8, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z10.b, p4, [x8, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z18.b, p4, [x24, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z18.b, p4, [x24, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z11.h, p0, [x23, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z11.h, p0, [x23, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z24.h, p3, [x1, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z24.h, p3, [x1, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z6.s, p5, [x23, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z6.s, p5, [x23, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z16.s, p6, [x14, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z16.s, p6, [x14, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z26.d, p2, [x7, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z26.d, p2, [x7, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z27.d, p1, [x12, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z27.d, p1, [x12, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Restricted predicate has range [0, 7].

st1b z12.b, p8, [x27, #6, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: st1b z12.b, p8, [x27, #6, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z23.h, p8, [x20, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: st1b z23.h, p8, [x20, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z6.s, p8, [x0, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: st1b z6.s, p8, [x0, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z14.d, p8, [x6, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: st1b z14.d, p8, [x6, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

st1b { }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: st1b { }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b { z1.b, z2.b }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1b { z1.b, z2.b }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b { v0.16b }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1b { v0.16b }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

st1b z0.b, p0, [x0, xzr]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: st1b z0.b, p0, [x0, xzr]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.b, p0, [x0, x0, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: st1b z0.b, p0, [x0, x0, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.b, p0, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: st1b z0.b, p0, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.b, p0, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: st1b z0.b, p0, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
