// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector lane index

bfmlslb z0.s, z0.h, z0.h[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: bfmlslb z0.s, z0.h, z0.h[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmlslb z0.s, z0.h, z0.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: bfmlslb z0.s, z0.h, z0.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmlslb z0.s, z0.h, z8.h[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfmlslb z0.s, z0.h, z8.h[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

bfmlslb z0.s, z0.s, z0.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmlslb z0.s, z0.s, z0.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmlslb z23.d, z23.h, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmlslb z23.d, z23.h, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmlslb z23.d, z23.h, z13.h[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmlslb z23.d, z23.h, z13.h[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
