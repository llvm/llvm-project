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
