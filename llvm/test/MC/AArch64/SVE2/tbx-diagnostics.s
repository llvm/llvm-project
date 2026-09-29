// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s

tbx z0.b, z1.b, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: tbx z0.b, z1.b, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
tbx  z31.d, z31.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: tbx  z31.d, z31.d, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
tbx  z31.d, z31.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: tbx  z31.d, z31.d, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
