// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

fcvt z0.h, {z0.s-z2.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvt z0.h, {z0.s-z2.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvt z0.h, {z1.s-z2.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element type
// CHECK-NEXT:  fcvt z0.h, {z1.s-z2.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

fcvt z0.s, {z0.s-z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvt z0.s, {z0.s-z1.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvt z0.h, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvt z0.h, {z0.h-z1.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
