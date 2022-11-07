// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

sqrshrun z0.h, {z0.s-z2.s}, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqrshrun z0.h, {z0.s-z2.s}, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrshrun z0.h, {z1.s-z2.s}, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: sqrshrun z0.h, {z1.s-z2.s}, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffixes

sqrshrun z0.b, {z0.s-z1.s}, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqrshrun z0.b, {z0.s-z1.s}, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrshrun z0.h, {z0.d-z1.d}, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqrshrun z0.h, {z0.d-z1.d}, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate

sqrshrun z0.h, {z0.s-z1.s}, #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16].
// CHECK-NEXT: sqrshrun z0.h, {z0.s-z1.s}, #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrshrun z0.h, {z0.s-z1.s}, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16].
// CHECK-NEXT: sqrshrun z0.h, {z0.s-z1.s}, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
