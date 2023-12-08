// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

sqcvtun z0.h, {z0.d-z4.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: sqcvtun z0.h, {z0.d-z4.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqcvtun z0.b, {z1.s-z4.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: sqcvtun z0.b, {z1.s-z4.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

sqcvtun z0.h, {z0.s-z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqcvtun z0.h, {z0.s-z3.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
