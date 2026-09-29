// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element size

cdot  z0.s, z1.h, z31.h, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cdot  z0.s, z1.h, z31.h, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cdot  z0.s, z1.s, z31.s, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cdot  z0.s, z1.s, z31.s, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cdot  z0.s, z1.d, z31.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cdot  z0.s, z1.d, z31.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cdot  z0.d, z1.b, z31.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cdot  z0.d, z1.b, z31.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cdot  z0.d, z1.s, z31.s, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cdot  z0.d, z1.s, z31.s, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cdot  z0.d, z1.d, z31.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cdot  z0.d, z1.d, z31.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid restricted register for indexed vector.

cdot  z0.s, z1.b, z8.b[3], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cdot  z0.s, z1.b, z8.b[3], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cdot  z0.d, z1.h, z16.h[1], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cdot  z0.d, z1.h, z16.h[1], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element index

cdot  z0.s, z1.b, z7.b[-1], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: cdot  z0.s, z1.b, z7.b[-1], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cdot  z0.s, z1.b, z7.b[4], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: cdot  z0.s, z1.b, z7.b[4], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cdot  z0.d, z1.h, z15.h[-1], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: cdot  z0.d, z1.h, z15.h[-1], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cdot  z0.d, z1.h, z15.h[2], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: cdot  z0.d, z1.h, z15.h[2], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid rotation

cdot z0.s, z1.b, z2.b[0], #360
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: complex rotation must be 0, 90, 180 or 270.
// CHECK-NEXT: cdot z0.s, z1.b, z2.b[0], #360
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cdot z0.d, z1.h, z2.h[0], #450
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: complex rotation must be 0, 90, 180 or 270.
// CHECK-NEXT: cdot z0.d, z1.h, z2.h[0], #450
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
cdot  z0.d, z1.h, z31.h, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: cdot  z0.d, z1.h, z31.h, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.d, p0/z, z7.d
cdot  z0.d, z1.h, z15.h[1], #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: cdot  z0.d, z1.h, z15.h[1], #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
