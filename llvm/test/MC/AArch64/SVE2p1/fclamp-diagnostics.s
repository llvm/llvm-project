// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector suffix

fclamp z23.h, z23.s, z13.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fclamp z23.h, z23.s, z13.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fclamp z23.s, z23.d, z13.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fclamp z23.s, z23.d, z13.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
