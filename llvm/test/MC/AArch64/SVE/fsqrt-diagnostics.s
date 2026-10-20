// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

fsqrt   z0.b, p0/m, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fsqrt   z0.b, p0/m, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsqrt   z0.s, p0/z, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: fsqrt   z0.s, p0/z, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsqrt   z0.s, p8/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fsqrt   z0.s, p8/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
