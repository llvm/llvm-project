// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+bitperm  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element width

bext z0.b, z1.h, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bext z0.b, z1.h, z2.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
bext z0.d, z1.d, z7.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: bext z0.d, z1.d, z7.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.d, p0/z, z7.d
bext z0.d, z1.d, z7.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: bext z0.d, z1.d, z7.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
