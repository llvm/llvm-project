// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Restricted predicate out of range.

fcmla z0.d, p8/m, z1.d, z2.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fcmla z0.d, p8/m, z1.d, z2.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid rotation

fcmla z0.d, p0/m, z1.d, z2.d, #360
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: complex rotation must be 0, 90, 180 or 270.
// CHECK-NEXT: fcmla z0.d, p0/m, z1.d, z2.d, #360
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcmla z0.d, p0/m, z1.d, z2.d, #450
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: complex rotation must be 0, 90, 180 or 270.
// CHECK-NEXT: fcmla z0.d, p0/m, z1.d, z2.d, #450
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Index out of bounds or invalid for element size

fcmla z0.h, z1.h, z2.h[-1], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fcmla z0.h, z1.h, z2.h[-1], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcmla z0.h, z1.h, z2.h[4], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fcmla z0.h, z1.h, z2.h[4], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcmla z0.s, z1.s, z2.s[-1], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: fcmla z0.s, z1.s, z2.s[-1], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcmla z0.s, z1.s, z2.s[2], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: fcmla z0.s, z1.s, z2.s[2], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcmla z0.d, z1.d, z2.d[0], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: fcmla z0.d, z1.d, z2.d[0], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z21.s, p0/z, z28.s
fcmla   z21.s, z10.s, z5.s[1], #90
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: fcmla   z21.s, z10.s, z5.s[1], #90
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
