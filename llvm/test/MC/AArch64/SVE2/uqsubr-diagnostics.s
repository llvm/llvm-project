// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Source and Destination Registers must match

uqsubr z0.b, p0/m, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: uqsubr z0.b, p0/m, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Element sizes must match

uqsubr z0.b, p0/m, z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: uqsubr z0.b, p0/m, z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsubr z0.b, p0/m, z0.b, z1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: uqsubr z0.b, p0/m, z0.b, z1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid predicate

uqsubr z0.b, p0/z, z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: uqsubr z0.b, p0/z, z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsubr z0.b, p8/m, z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: uqsubr z0.b, p8/m, z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
