// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of upper bound [-8, 7].

st1h z29.h, p5, [x7, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1h z29.h, p5, [x7, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1h z29.h, p5, [x4, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1h z29.h, p5, [x4, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1h z21.s, p2, [x1, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1h z21.s, p2, [x1, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1h z17.s, p5, [x1, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1h z17.s, p5, [x1, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1h z0.d, p1, [x14, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1h z0.d, p1, [x14, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1h z24.d, p3, [x16, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1h z24.d, p3, [x16, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Restricted predicate has range [0, 7].

st1h z15.h, p8, [x0, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: st1h z15.h, p8, [x0, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1h z17.s, p8, [x20, #2, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: st1h z17.s, p8, [x20, #2, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1h z15.d, p8, [x0, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: st1h z15.d, p8, [x0, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

st1h { }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: st1h { }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1h { z1.h, z2.h }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1h { z1.h, z2.h }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1h { v0.8h }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1h { v0.8h }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
