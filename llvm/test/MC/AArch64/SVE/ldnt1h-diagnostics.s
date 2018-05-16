// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of lower bound [-8, 7].

ldnt1h z23.h, p0/z, [x13, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ldnt1h z23.h, p0/z, [x13, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1h z29.h, p0/z, [x3, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ldnt1h z29.h, p0/z, [x3, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid result type.

ldnt1h z0.b, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ldnt1h z0.b, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1h z0.s, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ldnt1h z0.s, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1h z0.d, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ldnt1h z0.d, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

ldnt1h z27.h, p8/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ldnt1h z27.h, p8/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

ldnt1h { }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: ldnt1h { }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1h { z1.h, z2.h }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ldnt1h { z1.h, z2.h }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1h { v0.2d }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ldnt1h { v0.2d }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
