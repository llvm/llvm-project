// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

scvtf {z0.s-z3.s}, {z0.s-z4.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: scvtf {z0.s-z3.s}, {z0.s-z4.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtf {z1.s-z2.s}, {z0.s-z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: scvtf {z1.s-z2.s}, {z0.s-z1.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtf {z0.s-z3.s}, {z1.s-z5.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: scvtf {z0.s-z3.s}, {z1.s-z5.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

scvtf {z0.s-z3.s}, {z1.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: scvtf {z0.s-z3.s}, {z1.h-z3.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
