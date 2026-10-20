// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid element width

sqneg     z31.b, p7/z, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqneg     z31.b, p7/z, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqneg     z31.d, p7/z, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqneg     z31.d, p7/z, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid predicate

sqneg     z31.b, p8/z, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: sqneg     z31.b, p8/z, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.h, p0/z, z7.h
sqneg z0.h, p0/z, z3.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqneg z0.h, p0/z, z3.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
sqneg z0.h, p0/z, z3.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqneg z0.h, p0/z, z3.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: