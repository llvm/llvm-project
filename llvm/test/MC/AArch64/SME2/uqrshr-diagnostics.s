// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

uqrshr z0.b, {z0.s-z4.s}, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: uqrshr z0.b, {z0.s-z4.s}, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqrshr z0.h, {z10.s-z12.s}, #15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: uqrshr z0.h, {z10.s-z12.s}, #15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqrshr z0.h, {z1.d-z4.d}, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: uqrshr z0.h, {z1.d-z4.d}, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqrshr z0.h, {z1.s-z2.s}, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: uqrshr z0.h, {z1.s-z2.s}, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate

uqrshr z31.h, {z28.d-z31.d}, #65
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64].
// CHECK-NEXT: uqrshr z31.h, {z28.d-z31.d}, #65
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqrshr z31.h, {z28.s-z29.s}, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16].
// CHECK-NEXT: uqrshr z31.h, {z28.s-z29.s}, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqrshr z31.b, {z28.s-z31.s}, #33
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32].
// CHECK-NEXT: uqrshr z31.b, {z28.s-z31.s}, #33
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

uqrshr z23.s, {z12.s-z15.s}, #15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: uqrshr z23.s, {z12.s-z15.s}, #15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqrshr z23.b, {z12.d-z15.d}, #15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: uqrshr z23.b, {z12.d-z15.d}, #15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
