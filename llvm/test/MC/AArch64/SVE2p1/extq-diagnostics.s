// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid immediate

extq z23.b, z23.b, z13.b, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15].
// CHECK-NEXT: extq z23.b, z23.b, z13.b, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

extq z23.b, z23.b, z13.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15].
// CHECK-NEXT: extq z23.b, z23.b, z13.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

extq z23.h, z23.h, z13.h, #7
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: extq z23.h, z23.h, z13.h, #7
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
