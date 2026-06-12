/ RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid predicate

rbit  z0.b, p8/z, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: rbit  z0.b, p8/z, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

rbit  z0.h, p8/z, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: rbit  z0.h, p8/z, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

rbit  z0.s, p8/z, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: rbit  z0.s, p8/z, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

rbit  z0.d, p8/z, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: rbit  z0.d, p8/z, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element widths

rbit  z0.b, p7/z, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: rbit  z0.b, p7/z, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

rbit  z0.h, p7/z, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: rbit  z0.h, p7/z, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

rbit  z0.s, p7/z, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: rbit  z0.s, p7/z, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

rbit  z0.d, p7/z, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: rbit  z0.d, p7/z, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.b, p0/z, z7.b
rbit  z0.b, p0/z, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: rbit  z0.b, p0/z, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
rbit  z0.h, p0/z, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: rbit  z0.h, p0/z, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.s, p0/z, z7.s
rbit  z0.s, p0/z, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: rbit  z0.s, p0/z, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
rbit  z0.d, p0/z, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: rbit  z0.d, p0/z, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: