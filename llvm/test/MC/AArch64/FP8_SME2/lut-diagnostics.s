// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+sme-lutv2  2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector select register

luti4   {z0-z3}, zt0, {z0.b-z1.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti4   {z0-z3}, zt0, {z0.b-z1.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4   {z0.d, z4.d, z8.d, z12.d}, zt0, {z0-z1}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti4   {z0.d, z4.d, z8.d, z12.d}, zt0, {z0-z1}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector grouping

luti4   {z0.b-z1.b}, zt0, {z0-z4}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: luti4   {z0.b-z1.b}, zt0, {z0-z4}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4   {z0.b - z12.b}, zt0, {z0-z1}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: luti4   {z0.b - z12.b}, zt0, {z0-z1}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zip {z1.q-z2.q}, z0.q, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: zip {z1.q-z2.q}, z0.q, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zip {z1.q-z4.q}, z0.q, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: zip {z1.q-z4.q}, z0.q, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
