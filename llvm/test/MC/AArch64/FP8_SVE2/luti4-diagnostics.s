// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+lut  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid lane indices

luti4 z2.b, {z1.b}, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: luti4 z2.b, {z1.b}, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 z3.b, {z2.b}, z1[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: luti4 z3.b, {z2.b}, z1[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 z30.h, {z21.h}, z20[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti4 z30.h, {z21.h}, z20[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 z31.h, {z31.h}, z31[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti4 z31.h, {z31.h}, z31[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 z3.h, {z0.h, z1.h}, z2[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti4 z3.h, {z0.h, z1.h}, z2[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 z3.h, {z0.h, z1.h}, z2[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti4 z3.h, {z0.h, z1.h}, z2[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector lists

luti4 z30.h, {z0.h, z2.h}, z3[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti4 z30.h, {z0.h, z2.h}, z3[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

luti4 z2.h, {z1.b}, z0[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti4 z2.h, {z1.b}, z0[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 z31.h, {z31.b}, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti4 z31.h, {z31.b}, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 z3.s, {z0.h, z1.h}, z2[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: luti4 z3.s, {z0.h, z1.h}, z2[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
