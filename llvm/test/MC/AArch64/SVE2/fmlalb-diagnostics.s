// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// z register out of range for index

fmlalb z0.s, z1.h, z8.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlalb z0.s, z1.h, z8.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Index out of bounds

fmlalb z0.s, z1.h, z7.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: fmlalb z0.s, z1.h, z7.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalb z0.s, z1.h, z7.h[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: fmlalb z0.s, z1.h, z7.h[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid element width

fmlalb z0.s, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlalb z0.s, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalb z0.s, z1.s, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlalb z0.s, z1.s, z2.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalb z0.s, z1.d, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlalb z0.s, z1.d, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalb z0.s, z1.b, z2.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlalb z0.s, z1.b, z2.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalb z0.s, z1.s, z2.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlalb z0.s, z1.s, z2.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalb z0.s, z1.d, z2.d[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlalb z0.s, z1.d, z2.d[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z29.s, p0/z, z7.s
fmlalb  z29.s, z30.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: fmlalb  z29.s, z30.h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.s, p0/z, z7.s
fmlalb  z0.s, z1.h, z7.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: fmlalb  z0.s, z1.h, z7.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
