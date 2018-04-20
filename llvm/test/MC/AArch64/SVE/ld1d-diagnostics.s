// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of lower bound [-8, 7].

ld1d z28.d, p2/z, [x28, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1d z28.d, p2/z, [x28, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d z27.d, p1/z, [x26, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1d z27.d, p1/z, [x26, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

ld1d z4.d, p8/z, [x11, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld1d z4.d, p8/z, [x11, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

ld1d { }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: ld1d { }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d { z1.d, z2.d }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1d { z1.d, z2.d }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d { v0.2d }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1d { v0.2d }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

ld1d z0.d, p0/z, [x0, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1d z0.d, p0/z, [x0, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d z0.d, p0/z, [x0, xzr]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1d z0.d, p0/z, [x0, xzr]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d z0.d, p0/z, [x0, x0, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1d z0.d, p0/z, [x0, x0, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d z0.d, p0/z, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1d z0.d, p0/z, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d z0.d, p0/z, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1d z0.d, p0/z, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
