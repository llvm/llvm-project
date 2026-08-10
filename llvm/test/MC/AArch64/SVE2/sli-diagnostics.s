// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s

sli z18.b, z28.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7]
// CHECK-NEXT: sli z18.b, z28.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sli z1.b, z9.b, #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7]
// CHECK-NEXT: sli z1.b, z9.b, #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sli z21.h, z2.h, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15]
// CHECK-NEXT: sli z21.h, z2.h, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sli z14.h, z30.h, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15]
// CHECK-NEXT: sli z14.h, z30.h, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sli z6.s, z12.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31]
// CHECK-NEXT: sli z6.s, z12.s, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sli z23.s, z19.s, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31]
// CHECK-NEXT: sli z23.s, z19.s, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sli z3.d, z24.d, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63]
// CHECK-NEXT: sli z3.d, z24.d, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sli z25.d, z16.d, #64
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63]
// CHECK-NEXT: sli z25.d, z16.d, #64
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Element sizes must match

sli z0.b, z0.h, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sli z0.b, z0.h, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31, z6
sli     z31.d, z31.d, #63
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sli     z31.d, z31.d, #63
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
