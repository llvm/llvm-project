// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of lower bound [-8, 7].

ldnt1w z23.s, p0/z, [x13, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ldnt1w z23.s, p0/z, [x13, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1w z29.s, p0/z, [x3, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ldnt1w z29.s, p0/z, [x3, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid result type.

ldnt1w z0.b, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ldnt1w z0.b, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1w z0.h, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ldnt1w z0.h, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1w z0.d, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ldnt1w z0.d, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

ldnt1w z27.s, p8/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ldnt1w z27.s, p8/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

ldnt1w { }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: ldnt1w { }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1w { z1.s, z2.s }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ldnt1w { z1.s, z2.s }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1w { v0.2d }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ldnt1w { v0.2d }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.s, p0/z, z7.s
ldnt1w  { z0.s }, p0/z, [x0, x0, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ldnt1w  { z0.s }, p0/z, [x0, x0, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
ldnt1w  { z0.s }, p0/z, [x0, x0, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ldnt1w  { z0.s }, p0/z, [x0, x0, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
