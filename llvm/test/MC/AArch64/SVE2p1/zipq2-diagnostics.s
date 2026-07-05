// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid register suffixes

zipq2 z0.h, z0.h, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: zipq2 z0.h, z0.h, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zipq2 z0.d, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: zipq2 z0.d, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
