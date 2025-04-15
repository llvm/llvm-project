// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid element width

ursqrte     z31.b, p7/z, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ursqrte     z31.b, p7/z, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ursqrte     z31.h, p7/z, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ursqrte     z31.h, p7/z, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ursqrte     z31.s, p7/z, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ursqrte     z31.s, p7/z, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ursqrte     z31.d, p7/z, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ursqrte     z31.d, p7/z, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid predicate

ursqrte z0.s, p8/z, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: ursqrte z0.s, p8/z, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.s, p0/z, z7.s
ursqrte z0.s, p0/z, z3.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ursqrte z0.s, p0/z, z3.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
ursqrte z0.s, p0/z, z3.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ursqrte z0.s, p0/z, z3.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: