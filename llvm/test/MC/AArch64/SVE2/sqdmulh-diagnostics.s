// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// z register out of range for index

sqdmulh z0.h, z1.h, z8.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqdmulh z0.h, z1.h, z8.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.s, z1.s, z8.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqdmulh z0.s, z1.s, z8.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.d, z1.d, z16.d[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqdmulh z0.d, z1.d, z16.d[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element index

sqdmulh z0.h, z1.h, z2.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: sqdmulh z0.h, z1.h, z2.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.h, z1.h, z2.h[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: sqdmulh z0.h, z1.h, z2.h[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.s, z1.s, z2.s[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: sqdmulh z0.s, z1.s, z2.s[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.s, z1.s, z2.s[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: sqdmulh z0.s, z1.s, z2.s[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.d, z1.d, z2.d[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: sqdmulh z0.d, z1.d, z2.d[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.d, z1.d, z2.d[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: sqdmulh z0.d, z1.d, z2.d[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid element width

sqdmulh z0.b, z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqdmulh z0.b, z1.h, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.h, z1.s, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqdmulh z0.h, z1.s, z2.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.s, z1.d, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqdmulh z0.s, z1.d, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.d, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqdmulh z0.d, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.b, z1.b, z2.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqdmulh z0.b, z1.b, z2.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.b, z1.h, z2.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqdmulh z0.b, z1.h, z2.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.h, z1.s, z2.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqdmulh z0.h, z1.s, z2.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.s, z1.d, z2.d[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqdmulh z0.s, z1.d, z2.d[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh z0.d, z1.b, z2.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqdmulh z0.d, z1.b, z2.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
sqdmulh    z31.d, z31.d, z15.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqdmulh    z31.d, z31.d, z15.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
sqdmulh    z31.d, z31.d, z15.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqdmulh    z31.d, z31.d, z15.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31.d, p0/z, z6.d
sqdmulh    z31.d, z31.d, z15.d[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqdmulh    z31.d, z31.d, z15.d[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
sqdmulh    z31.d, z31.d, z15.d[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqdmulh    z31.d, z31.d, z15.d[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
