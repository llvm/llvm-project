// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2-aes  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element width

pmullb z0.q, z0.q, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: pmullb z0.q, z0.q, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
pmullb z0.q, z1.d, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: pmullb z0.q, z1.d, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
pmullb z0.q, z1.d, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: pmullb z0.q, z1.d, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
