// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s

sshllt z18.h, z28.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7]
// CHECK-NEXT: sshllt z18.h, z28.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sshllt z1.h, z9.b, #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7]
// CHECK-NEXT: sshllt z1.h, z9.b, #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sshllt z21.s, z2.h, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15]
// CHECK-NEXT: sshllt z21.s, z2.h, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sshllt z14.s, z30.h, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15]
// CHECK-NEXT: sshllt z14.s, z30.h, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sshllt z6.d, z12.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31]
// CHECK-NEXT: sshllt z6.d, z12.s, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sshllt z23.d, z19.s, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31]
// CHECK-NEXT: sshllt z23.d, z19.s, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid element width

sshllt z0.b, z0.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sshllt z0.b, z0.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sshllt z0.h, z0.h, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sshllt z0.h, z0.h, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sshllt z0.s, z0.s, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sshllt z0.s, z0.s, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sshllt z0.d, z0.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sshllt z0.d, z0.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31, z6
sshllt     z31.d, z31.s, #31
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sshllt     z31.d, z31.s, #31
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31.d, p0/m, z6.d
sshllt     z31.d, z31.s, #31
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sshllt     z31.d, z31.s, #31
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
