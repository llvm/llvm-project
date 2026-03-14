// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid predicate

sqabs z0.s, p0/z, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: sqabs z0.s, p0/z, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqabs z0.s, p8/m, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: sqabs z0.s, p8/m, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element width

sqabs z0.b, p7/m, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqabs z0.b, p7/m, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
