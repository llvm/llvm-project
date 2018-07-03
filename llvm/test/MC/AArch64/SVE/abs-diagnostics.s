// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Element size specifiers should match.
abs z0.h, p0/m, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: abs z0.h, p0/m, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Missing predicate suffix
abs z29.d, p7, z29.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: abs z29.d, p7, z29.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// error: restricted predicate has range [0, 7].

abs z0.b, p8/m, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: abs z0.b, p8/m, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

abs z0.h, p8/m, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: abs z0.h, p8/m, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

abs z0.s, p8/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: abs z0.s, p8/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

abs z0.d, p8/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: abs z0.d, p8/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
