// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3,+fpv2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid predicate register

fcmtl p0.b, z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: fcmtl p0.b, z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcmtlmg p0.b, z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: fcmtlmg p0.b, z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element widths

fcmtl p0.h, z0.s, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcmtl p0.h, z0.s, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcmtl p0.s, z0.s, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcmtl p0.s, z0.s, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcmtlmg p0.h, z0.s, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcmtlmg p0.h, z0.s, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcmtlmg p0.s, z0.s, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcmtlmg p0.s, z0.s, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
fcmtl p0.d, z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcmtl p0.d, z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
fcmtl p0.d, z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcmtl p0.d, z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.d, p0/z, z7.d
fcmtlmg p0.d, z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcmtlmg p0.d, z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
fcmtlmg p0.d, z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcmtlmg p0.d, z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
