// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

sunpk {z0.h-z2.h}, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sunpk {z0.h-z2.h}, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sunpk {z1.s-z2.s}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: sunpk {z1.s-z2.s}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sunpk {z0.d-z5.d}, {z8.s-z9.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: sunpk {z0.d-z5.d}, {z8.s-z9.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sunpk {z0.s-z3.s}, {z9.h-z11.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sunpk {z0.s-z3.s}, {z9.h-z11.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

sunpk {z0.s-z3.s}, {z8.s-z9.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sunpk {z0.s-z3.s}, {z8.s-z9.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
