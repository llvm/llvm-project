// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Out of range index offset

svdot za.s[w8, 8, vgx4], {z0.b-z3.b}, z0.b[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: svdot za.s[w8, 8, vgx4], {z0.b-z3.b}, z0.b[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

svdot za.s[w8, -1, vgx4], {z0.b-z3.b}, z0.b[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: svdot za.s[w8, -1, vgx4], {z0.b-z3.b}, z0.b[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

svdot za.s[w7, 7, vgx2], {z0.h-z1.h}, z0.h[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: svdot za.s[w7, 7, vgx2], {z0.h-z1.h}, z0.h[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

svdot za.d[w8, 0, vgx4], {z0.h-z4.h}, z0.h[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: svdot za.d[w8, 0, vgx4], {z0.h-z4.h}, z0.h[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

svdot za.s[w8, 0, vgx4], {z1.b-z4.b}, z0.b[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: svdot za.s[w8, 0, vgx4], {z1.b-z4.b}, z0.b[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Matrix Operand

svdot za.b[w8, 0, vgx4], {z0.h-z3.h}, z4.h[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: svdot za.b[w8, 0, vgx4], {z0.h-z3.h}, z4.h[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector grouping

svdot za.s[w8, 0, vgx2], {z0.b-z3.b}, z14.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: svdot za.s[w8, 0, vgx2], {z0.b-z3.b}, z14.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
