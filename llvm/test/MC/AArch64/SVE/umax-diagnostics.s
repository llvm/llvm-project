// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

umax    z0.b, z0.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255].
// CHECK-NEXT: umax    z0.b, z0.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umax    z31.b, z31.b, #256
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255].
// CHECK-NEXT: umax    z31.b, z31.b, #256
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umax    z0.b, p8/m, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: umax    z0.b, p8/m, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
