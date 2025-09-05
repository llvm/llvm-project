// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid element width

scvtf z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtf z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtf z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtf z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtf z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtf z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtf z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtf z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
scvtf z0.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: scvtf z0.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element width

scvtflt z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtflt z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtflt z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtflt z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtflt z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtflt z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtflt z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtflt z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
scvtflt z0.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: scvtflt z0.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
