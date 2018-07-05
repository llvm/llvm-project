// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Source and Destination Registers must match

fcadd z0.d, p2/m, z1.d, z2.d, #90
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: fcadd z0.d, p2/m, z1.d, z2.d, #90
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Restricted predicate out of range.

fcadd z0.d, p8/m, z0.d, z1.d, #90
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: fcadd z0.d, p8/m, z0.d, z1.d, #90
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid rotation

fcadd z0.d, p0/m, z0.d, z1.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: complex rotation must be 90 or 270.
// CHECK-NEXT: fcadd z0.d, p0/m, z0.d, z1.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcadd z0.d, p0/m, z0.d, z1.d, #180
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: complex rotation must be 90 or 270.
// CHECK-NEXT: fcadd z0.d, p0/m, z0.d, z1.d, #180
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcadd z0.d, p0/m, z0.d, z1.d, #450
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: complex rotation must be 90 or 270.
// CHECK-NEXT: fcadd z0.d, p0/m, z0.d, z1.d, #450
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
