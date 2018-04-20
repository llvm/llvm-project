// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid operand (.h)

ld1sh z23.h, p0/z, [x13, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1sh z23.h, p0/z, [x13, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1sh z29.h, p0/z, [x3, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1sh z29.h, p0/z, [x3, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Immediate out of lower bound [-8, 7].

ld1sh z30.s, p6/z, [x25, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1sh z30.s, p6/z, [x25, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1sh z29.s, p5/z, [x15, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1sh z29.s, p5/z, [x15, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1sh z28.d, p2/z, [x28, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1sh z28.d, p2/z, [x28, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1sh z27.d, p1/z, [x26, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1sh z27.d, p1/z, [x26, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

ld1sh z12.s, p8/z, [x13, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld1sh z12.s, p8/z, [x13, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1sh z4.d, p8/z, [x11, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld1sh z4.d, p8/z, [x11, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

ld1sh { }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: ld1sh { }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1sh { z1.s, z2.s }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1sh { z1.s, z2.s }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1sh { v0.2d }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1sh { v0.2d }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

ld1sh z0.s, p0/z, [x0, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1sh z0.s, p0/z, [x0, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1sh z0.s, p0/z, [x0, xzr]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1sh z0.s, p0/z, [x0, xzr]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1sh z0.s, p0/z, [x0, x0, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1sh z0.s, p0/z, [x0, x0, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1sh z0.s, p0/z, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1sh z0.s, p0/z, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1sh z0.s, p0/z, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1sh z0.s, p0/z, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
