// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2,+f16mm 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid element width

fmmla z0.b, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmmla z0.b, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
