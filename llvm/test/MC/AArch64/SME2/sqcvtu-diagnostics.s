// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

sqcvtu z0.h, {z0.s-z2.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqcvtu z0.h, {z0.s-z2.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqcvtu z0.b, {z1.s-z4.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element type
// CHECK-NEXT: sqcvtu z0.b, {z1.s-z4.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqcvtu z0.h, {z1.d-z2.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqcvtu z0.h, {z1.d-z2.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

sqcvtu z0.s, {z0.s-z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqcvtu z0.s, {z0.s-z1.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
