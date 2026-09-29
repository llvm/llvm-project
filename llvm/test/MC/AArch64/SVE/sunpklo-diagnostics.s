// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element widths.

sunpklo z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sunpklo z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sunpklo z0.s, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sunpklo z0.s, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sunpklo z0.d, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sunpklo z0.d, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
sunpklo z31.d, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sunpklo z31.d, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
sunpklo z31.d, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sunpklo z31.d, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
