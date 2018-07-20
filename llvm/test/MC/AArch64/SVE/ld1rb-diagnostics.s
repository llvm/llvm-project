// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of lower bound [0, 63].

ld1rb z0.b, p1/z, [x0, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be in range [0, 63].
// CHECK-NEXT: ld1rb z0.b, p1/z, [x0, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rb z0.b, p1/z, [x0, #64]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be in range [0, 63].
// CHECK-NEXT: ld1rb z0.b, p1/z, [x0, #64]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

ld1rb z0.b, p8/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld1rb z0.b, p8/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
