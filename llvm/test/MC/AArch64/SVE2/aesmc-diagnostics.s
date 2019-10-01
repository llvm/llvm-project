// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2-aes  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Source and Destination Registers must match

aesmc z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: aesmc z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid element width

aesmc z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: aesmc z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesmc z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: aesmc z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesmc z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: aesmc z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.b, p0/z, z7.b
aesmc z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: aesmc z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
aesmc z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: aesmc z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
