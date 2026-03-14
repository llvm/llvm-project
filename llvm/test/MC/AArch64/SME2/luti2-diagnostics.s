// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid lane indices

luti2 z0.h, zt0, z0[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: luti2 z0.h, zt0, z0[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 z0.s, zt0, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: luti2 z0.s, zt0, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z0.b-z1.b}, zt0, z0[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: luti2 {z0.b-z1.b}, zt0, z0[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z0.h-z1.h}, zt0, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: luti2 {z0.h-z1.h}, zt0, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z0.s-z3.s}, zt0, z0[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti2 {z0.s-z3.s}, zt0, z0[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z0.b-z3.b}, zt0, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti2 {z0.b-z3.b}, zt0, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector lists

luti2 {z0.h-z2.h}, zt0, z0[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti2 {z0.h-z2.h}, zt0, z0[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z1.s-z2.s}, zt0, z0[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: luti2 {z1.s-z2.s}, zt0, z0[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 {z1.s-z4.s}, zt0, z0[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: luti2 {z1.s-z4.s}, zt0, z0[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

luti2 {z0.d-z1.d}, zt0, z0[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti2 {z0.d-z1.d}, zt0, z0[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
