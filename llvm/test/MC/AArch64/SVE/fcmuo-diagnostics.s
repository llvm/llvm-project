// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

fcmuo   p0.b, p0/z, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: fcmuo   p0.b, p0/z, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcmuo   p0.s, p0/z, z0.s, #0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected floating point literal
// CHECK-NEXT: fcmuo   p0.s, p0/z, z0.s, #0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
