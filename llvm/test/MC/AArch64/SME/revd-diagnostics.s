// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid predicate

// invalid range (expected: p0-p7)
revd z0.q, p8/m, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: revd z0.q, p8/m, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// wrong predication qualifier, expected /m.
revd z0.q, p0/z, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: revd z0.q, p0/z, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid ZPR element width

revd z0.b, p0/m, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: revd z0.b, p0/m, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

revd z0.q, p0/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: revd z0.q, p0/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z21, z25
revd    z21.q, p5/m, z10.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: revd    z21.q, p5/m, z10.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
