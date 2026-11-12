// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2-aes  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Source and Destination Registers must match

aesimc z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: aesimc z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid element width

aesimc z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: aesimc z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesimc z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: aesimc z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesimc z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: aesimc z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.b, p0/z, z7.b
aesimc z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: aesimc z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
aesimc z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: aesimc z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
