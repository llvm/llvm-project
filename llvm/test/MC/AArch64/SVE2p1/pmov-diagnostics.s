// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector lane indices

pmov p0.b, z0[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected lane specifier '[0]'
// CHECK-NEXT: pmov p0.b, z0[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmov p0.h, z0[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: pmov p0.h, z0[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmov p0.h, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: pmov p0.h, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmov p0.s, z0[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: pmov p0.s, z0[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmov p0.s, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: pmov p0.s, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmov p0.d, z0[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: pmov p0.d, z0[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmov p0.d, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: pmov p0.d, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:



pmov z0[2], p0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: pmov z0[2], p0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmov z0[-1], p0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: pmov z0[-1], p0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmov z0[4], p0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: pmov z0[4], p0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmov z0[-1], p0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: pmov z0[-1], p0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmov z0[8], p0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: pmov z0[8], p0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pmov z0[-1], p0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: pmov z0[-1], p0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

