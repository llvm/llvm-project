// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid element width

ucvtf z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtf z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ucvtf z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtf z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ucvtf z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtf z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ucvtf z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtf z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
ucvtf z0.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ucvtf z0.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element width

ucvtflt z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtflt z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ucvtflt z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtflt z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ucvtflt z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtflt z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ucvtflt z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtflt z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
ucvtflt z0.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ucvtflt z0.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
