// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

smax    z0.b, z0.b, #-129
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-128, 127].
// CHECK-NEXT: smax    z0.b, z0.b, #-129
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smax    z31.b, z31.b, #128
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-128, 127].
// CHECK-NEXT: smax    z31.b, z31.b, #128
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smax    z0.b, p8/m, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: smax    z0.b, p8/m, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
