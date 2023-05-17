// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid lane indices

luti2 {z0.h, z8.h}, zt0, z0[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: luti2 {z0.h, z8.h}, zt0, z0[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z0.h, z8.h}, zt0, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: luti2 {z0.h, z8.h}, zt0, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z0.h, z8.h}, zt0, z0[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: luti2 {z0.h, z8.h}, zt0, z0[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z0.h, z8.h}, zt0, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: luti2 {z0.h, z8.h}, zt0, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z0.b, z8.b}, zt0, z0[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: luti2 {z0.b, z8.b}, zt0, z0[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z0.b, z8.b}, zt0, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: luti2 {z0.b, z8.b}, zt0, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z19.h, z23.h, z27.h, z31.h}, zt0, z31[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti2 {z19.h, z23.h, z27.h, z31.h}, zt0, z31[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z19.h, z23.h, z27.h, z31.h}, zt0, z31[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti2 {z19.h, z23.h, z27.h, z31.h}, zt0, z31[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z19.b, z23.b, z27.b, z31.b}, zt0, z31[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti2 {z19.b, z23.b, z27.b, z31.b}, zt0, z31[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z19.b, z23.b, z27.b, z31.b}, zt0, z31[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti2 {z19.b, z23.b, z27.b, z31.b}, zt0, z31[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector lists

luti2 {z0.h, z9.h}, zt0, z0[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti2 {z0.h, z9.h}, zt0, z0[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

luti2 {z0.d, z2.d}, zt0, z0[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti2 {z0.d, z2.d}, zt0, z0[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
