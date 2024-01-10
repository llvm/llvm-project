// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector lane index

fdot z0.s, z0.h, z0.h[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fdot z0.s, z0.h, z0.h[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot z0.s, z0.h, z0.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fdot z0.s, z0.h, z0.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

fdot z0.h, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot z0.h, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot z0.d, z0.h, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fdot z0.d, z0.h, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
