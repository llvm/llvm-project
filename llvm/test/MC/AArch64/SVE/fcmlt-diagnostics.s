// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

fcmlt   p0.b, p0/z, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: fcmlt   p0.b, p0/z, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcmlt   p0.s, p0/z, z0.s, #1.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected floating-point constant #0.0
// CHECK-NEXT: fcmlt   p0.s, p0/z, z0.s, #1.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
fcmlt   p0.d, p0/z, z0.d, #0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcmlt   p0.d, p0/z, z0.d, #0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
fcmlt   p0.d, p0/z, z0.d, #0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcmlt   p0.d, p0/z, z0.d, #0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.d, p0/z, z7.d
fcmlt   p0.d, p0/z, z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcmlt   p0.d, p0/z, z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
fcmlt   p0.d, p0/z, z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcmlt   p0.d, p0/z, z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
