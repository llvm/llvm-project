// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve-b16mm 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid element width

bfmmla z0.h, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmmla z0.h, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmmla z0.s, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmmla z0.s, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmmla z0.d, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmmla z0.d, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.h, p0/m, z7.h
bfmmla z0.h, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx and destination also used as non-destructive source
// CHECK-NEXT: bfmmla z0.h, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
bfmmla z0.h, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx and destination also used as non-destructive source
// CHECK-NEXT: bfmmla z0.h, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
