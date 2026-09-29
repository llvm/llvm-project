// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid lane indices

luti4 {z0.h, z8.h}, zt0, z0[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti4 {z0.h, z8.h}, zt0, z0[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 {z0.h, z8.h}, zt0, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti4 {z0.h, z8.h}, zt0, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 {z0.h, z8.h}, zt0, z0[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti4 {z0.h, z8.h}, zt0, z0[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 {z0.h, z8.h}, zt0, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti4 {z0.h, z8.h}, zt0, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 {z0.b, z8.b}, zt0, z0[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti4 {z0.b, z8.b}, zt0, z0[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 {z0.b, z8.b}, zt0, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti4 {z0.b, z8.b}, zt0, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 {z19.h, z23.h, z27.h, z31.h}, zt0, z31[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: luti4 {z19.h, z23.h, z27.h, z31.h}, zt0, z31[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 {z19.h, z23.h, z27.h, z31.h}, zt0, z31[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: luti4 {z19.h, z23.h, z27.h, z31.h}, zt0, z31[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 {z19.h, z23.h, z27.h, z31.h}, zt0, z31[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: luti4 {z19.h, z23.h, z27.h, z31.h}, zt0, z31[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 {z19.h, z23.h, z27.h, z31.h}, zt0, z31[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: luti4 {z19.h, z23.h, z27.h, z31.h}, zt0, z31[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector lists

luti4 {z1.s-z4.s}, zt0, z0[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: luti4 {z1.s-z4.s}, zt0, z0[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

