// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+lut  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid lane indices

luti2 z2.b, {z1.b}, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti2 z2.b, {z1.b}, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 z3.b, {z2.b}, z1[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti2 z3.b, {z2.b}, z1[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 z30.h, {z21.h}, z20[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: luti2 z30.h, {z21.h}, z20[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 z31.h, {z31.h}, z31[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: luti2 z31.h, {z31.h}, z31[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

luti2 z2.h, {z1.b}, z0[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti2 z2.h, {z1.b}, z0[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 z31.b, {z31.h}, z31[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: luti2 z31.b, {z31.h}, z31[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
