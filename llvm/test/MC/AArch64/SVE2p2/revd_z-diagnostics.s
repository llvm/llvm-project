// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid predicate

revd  z0.q, p8/z, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: revd  z0.q, p8/z, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element widths

revd  z0.b, p7/z, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: revd  z0.b, p7/z, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

revd  z0.h, p7/z, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: revd  z0.h, p7/z, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

revd  z0.s, p7/z, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: revd  z0.s, p7/z, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

revd  z0.h, p7/z, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: revd  z0.h, p7/z, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

revd  z0.s, p7/z, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: revd  z0.s, p7/z, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

revd  z0.d, p7/z, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: revd  z0.d, p7/z, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

revd  z0.q, p7/z, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: revd  z0.q, p7/z, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
revd  z0.q, p0/z, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: revd  z0.q, p0/z, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: