// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

fminnm {z0.d, z1.d}, {z0.d-z2.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fminnm {z0.d, z1.d}, {z0.d-z2.d}, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fminnm {z1.s-z2.s}, {z0.s, z1.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element type
// CHECK-NEXT: fminnm {z1.s-z2.s}, {z0.s, z1.s}, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid single register

fminnm {z0.h, z1.h}, {z2.h-z3.h}, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: fminnm {z0.h, z1.h}, {z2.h-z3.h}, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

fminnm {z0.h, z1.h}, {z2.h-z3.h}, z14.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: fminnm {z0.h, z1.h}, {z2.h-z3.h}, z14.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
