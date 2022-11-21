// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1,+b16b16 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector suffix

bfclamp z23.h, z23.s, z13.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfclamp z23.h, z23.s, z13.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfclamp z23.s, z23.h, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfclamp z23.s, z23.h, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
