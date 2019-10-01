// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Source and Destination Registers must match

cadd z0.d, z1.d, z2.d, #90
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: cadd z0.d, z1.d, z2.d, #90
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid rotation

cadd z0.d, z0.d, z1.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: complex rotation must be 90 or 270.
// CHECK-NEXT: cadd z0.d, z0.d, z1.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cadd z0.d, z0.d, z1.d, #180
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: complex rotation must be 90 or 270.
// CHECK-NEXT: cadd z0.d, z0.d, z1.d, #180
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cadd z0.d, z0.d, z1.d, #450
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: complex rotation must be 90 or 270.
// CHECK-NEXT: cadd z0.d, z0.d, z1.d, #450
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
cadd z0.d, z0.d, z31.d, #90
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: cadd z0.d, z0.d, z31.d, #90
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
