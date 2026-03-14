// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s

urshr z18.b, p0/m, z18.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8]
// CHECK-NEXT: urshr z18.b, p0/m, z18.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshr z1.b, p0/m, z1.b, #9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8]
// CHECK-NEXT: urshr z1.b, p0/m, z1.b, #9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshr z21.h, p0/m, z21.h, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: urshr z21.h, p0/m, z21.h, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshr z14.h, p0/m, z14.h, #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: urshr z14.h, p0/m, z14.h, #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshr z6.s, p0/m, z6.s, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32]
// CHECK-NEXT: urshr z6.s, p0/m, z6.s, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshr z23.s, p0/m, z23.s, #33
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32]
// CHECK-NEXT: urshr z23.s, p0/m, z23.s, #33
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshr z3.d, p0/m, z3.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]
// CHECK-NEXT: urshr z3.d, p0/m, z3.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshr z25.d, p0/m, z25.d, #65
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]
// CHECK-NEXT: urshr z25.d, p0/m, z25.d, #65
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Source and Destination Registers must match

urshr z0.b, p0/m, z1.b, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: urshr z0.b, p0/m, z1.b, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Element sizes must match

urshr z0.b, p0/m, z0.d, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: urshr z0.b, p0/m, z0.d, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshr z0.d, p0/m, z0.b, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: urshr z0.d, p0/m, z0.b, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid predicate

urshr z0.b, p0/z, z0.b, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: urshr z0.b, p0/z, z0.b, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshr z0.b, p8/m, z0.b, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: urshr z0.b, p8/m, z0.b, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
