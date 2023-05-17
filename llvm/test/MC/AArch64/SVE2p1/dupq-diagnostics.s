// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector lane index

dupq z0.b, z0.b[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: dupq z0.b, z0.b[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dupq z0.b, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: dupq z0.b, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dupq z0.h, z0.h[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: dupq z0.h, z0.h[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dupq z0.h, z0.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: dupq z0.h, z0.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dupq z0.s, z0.s[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: dupq z0.s, z0.s[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dupq z0.s, z0.s[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: dupq z0.s, z0.s[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dupq z0.d, z0.d[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: dupq z0.d, z0.d[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dupq z0.d, z0.d[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: dupq z0.d, z0.d[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

dupq z0.s, z0.h[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: dupq z0.s, z0.h[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
