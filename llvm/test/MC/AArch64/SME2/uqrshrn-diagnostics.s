// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

uqrshrn z0.b, {z0.s-z4.s}, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: uqrshrn z0.b, {z0.s-z4.s}, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqrshrn z0.h, {z1.d-z4.d}, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: uqrshrn z0.h, {z1.d-z4.d}, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate

uqrshrn z31.h, {z28.d-z31.d}, #65
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64].
// CHECK-NEXT: uqrshrn z31.h, {z28.d-z31.d}, #65
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

uqrshrn z23.s, {z12.s-z15.s}, #24
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: uqrshrn z23.s, {z12.s-z15.s}, #24
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
