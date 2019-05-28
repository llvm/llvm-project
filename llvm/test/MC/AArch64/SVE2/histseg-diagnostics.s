// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Invalid element width

histseg z0.h, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: histseg z0.h, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

histseg z0.s, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: histseg z0.s, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

histseg z0.d, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: histseg z0.d, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31, z6
histseg z31.b, z30.b, z29.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: histseg z31.b, z30.b, z29.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31.b, p0/m, z6.b
histseg z31.b, z30.b, z29.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: histseg z31.b, z30.b, z29.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
