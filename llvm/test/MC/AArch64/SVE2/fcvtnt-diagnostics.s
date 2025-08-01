// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Invalid element width

fcvtnt z0.b, p0/m, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtnt z0.b, p0/m, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtnt z0.h, p0/m, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcvtnt z0.h, p0/m, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtnt z0.s, p0/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcvtnt z0.s, p0/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtnt z0.d, p0/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcvtnt z0.d, p0/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtnt z0.b, p0/m, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtnt z0.b, p0/m, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtnt z0.d, p0/m, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcvtnt z0.d, p0/m, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid predicate operation

fcvtnt z0.h, p0/z, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: fcvtnt z0.h, p0/z, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Predicate not in restricted predicate range

fcvtnt z0.h, p8/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fcvtnt z0.h, p8/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.s, p0/m, z7.s
fcvtnt z0.s, p7/m, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcvtnt z0.s, p7/m, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
fcvtnt z0.s, p7/m, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcvtnt z0.s, p7/m, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
