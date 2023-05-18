// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

umax {z0.h, z1.h}, {z0.h-z2.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: umax {z0.h, z1.h}, {z0.h-z2.h}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umax {z1.d-z2.d}, {z0.d, z1.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element type
// CHECK-NEXT: umax {z1.d-z2.d}, {z0.d, z1.d}, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid single register

umax {z0.b, z1.b}, {z2.b-z3.b}, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: umax {z0.b, z1.b}, {z2.b-z3.b}, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

umax {z0.b, z1.b}, {z2.b-z3.b}, z14.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: umax {z0.b, z1.b}, {z2.b-z3.b}, z14.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
