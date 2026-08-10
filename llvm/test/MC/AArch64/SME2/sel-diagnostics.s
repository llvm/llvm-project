// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

sel {z0.h-z2.h}, pn8, {z0.h-z1.h}, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sel {z0.h-z2.h}, pn8, {z0.h-z1.h}, {z0.h-z1.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sel {z28.s-z31.s}, pn15, {z26.s-z31.s}, {z28.s-z31.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: sel {z28.s-z31.s}, pn15, {z26.s-z31.s}, {z28.s-z31.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sel {z28.d-z31.d}, pn15, {z28.d-z31.d}, {z26.d-z31.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: sel {z28.d-z31.d}, pn15, {z28.d-z31.d}, {z26.d-z31.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sel {z1.b-z4.b}, pn8, {z0.b-z3.b}, {z0.b-z3.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: sel {z1.b-z4.b}, pn8, {z0.b-z3.b}, {z0.b-z3.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sel {z22.s-z23.s}, pn11, {z13.s-z14.s}, {z8.s-z9.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: sel {z22.s-z23.s}, pn11, {z13.s-z14.s}, {z8.s-z9.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

sel {z0.h-z1.h}, pn8, {z0.h-z1.h}, {z0.s-z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sel {z0.h-z1.h}, pn8, {z0.h-z1.h}, {z0.s-z1.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
