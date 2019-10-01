// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element width

sqxtunt z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqxtunt z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqxtunt z0.h, z1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqxtunt z0.h, z1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqxtunt z0.s, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqxtunt z0.s, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqxtunt z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqxtunt z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
sqxtunt  z0.s, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqxtunt  z0.s, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.s, p0/z, z7.s
sqxtunt  z0.s, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqxtunt  z0.s, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
