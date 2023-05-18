// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element width

bcax z0.b, z0.b, z1.s, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bcax z0.b, z0.b, z1.s, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bcax z0.h, z0.h, z1.h, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bcax z0.h, z0.h, z1.h, z2.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bcax z0.d, z0.d, z1.s, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bcax z0.d, z0.d, z1.s, z2.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Source and Destination Registers must match

bcax z0.d, z1.d, z2.d, z3.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: bcax z0.d, z1.d, z2.d, z3.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
bcax z0.d, z0.d, z1.d, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: bcax z0.d, z0.d, z1.d, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
