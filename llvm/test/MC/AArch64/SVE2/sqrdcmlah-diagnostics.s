// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element size

sqrdcmlah  z0.h, z1.b, z2.b[0], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqrdcmlah  z0.h, z1.b, z2.b[0], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrdcmlah  z0.h, z1.s, z2.s[0], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqrdcmlah  z0.h, z1.s, z2.s[0], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrdcmlah  z0.h, z1.d, z2.d[0], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqrdcmlah  z0.h, z1.d, z2.d[0], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrdcmlah  z0.s, z1.b, z2.b[0], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqrdcmlah  z0.s, z1.b, z2.b[0], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrdcmlah  z0.s, z1.h, z2.h[0], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqrdcmlah  z0.s, z1.h, z2.h[0], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrdcmlah  z0.s, z1.d, z2.d[0], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqrdcmlah  z0.s, z1.d, z2.d[0], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid restricted register for indexed vector.

sqrdcmlah  z0.h, z1.h, z8.h[3], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqrdcmlah  z0.h, z1.h, z8.h[3], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrdcmlah  z0.s, z1.s, z16.s[1], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqrdcmlah  z0.s, z1.s, z16.s[1], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element index

sqrdcmlah  z0.h, z1.h, z7.h[-1], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: sqrdcmlah  z0.h, z1.h, z7.h[-1], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrdcmlah  z0.h, z1.h, z7.h[4], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: sqrdcmlah  z0.h, z1.h, z7.h[4], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrdcmlah  z0.s, z1.s, z15.s[-1], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: sqrdcmlah  z0.s, z1.s, z15.s[-1], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrdcmlah  z0.s, z1.s, z15.s[2], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: sqrdcmlah  z0.s, z1.s, z15.s[2], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid rotation

sqrdcmlah z0.h, z1.h, z2.h[0], #360
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: complex rotation must be 0, 90, 180 or 270.
// CHECK-NEXT: sqrdcmlah z0.h, z1.h, z2.h[0], #360
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrdcmlah z0.s, z1.s, z2.s[0], #450
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: complex rotation must be 0, 90, 180 or 270.
// CHECK-NEXT: sqrdcmlah z0.s, z1.s, z2.s[0], #450
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.b, p0/z, z7.b
sqrdcmlah  z0.b, z1.b, z31.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: sqrdcmlah  z0.b, z1.b, z31.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.s, p0/z, z7.s
sqrdcmlah  z0.s, z1.s, z15.s[1], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: sqrdcmlah  z0.s, z1.s, z15.s[1], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
