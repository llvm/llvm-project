// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s
// --------------------------------------------------------------------------//
// Out of range index offset

usvdot za.s[w8, 8, vgx4], {z0.b-z3.b}, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: usvdot za.s[w8, 8, vgx4], {z0.b-z3.b}, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

usvdot za.s[w8, -1, vgx4], {z0.b-z3.b}, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: usvdot za.s[w8, -1, vgx4], {z0.b-z3.b}, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

usvdot za.s[w7, 0, vgx4], {z4.b-z7.b}, z0.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: usvdot za.s[w7, 0, vgx4], {z4.b-z7.b}, z0.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

usvdot za.s[w12, 0, vgx2], {z4.b-z5.b}, z0.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: usvdot za.s[w12, 0, vgx2], {z4.b-z5.b}, z0.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

usvdot za.s[w8, 0, vgx4], {z0.b-z4.b}, z0.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: usvdot za.s[w8, 0, vgx4], {z0.b-z4.b}, z0.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

usvdot za.s[w8, 0, vgx4], {z1.b-z4.b}, z15.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: usvdot za.s[w8, 0, vgx4], {z1.b-z4.b}, z15.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Matrix Operand

usvdot za.h[w8, 0, vgx4], {z0.b-z3.b}, z4.b[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: usvdot za.h[w8, 0, vgx4], {z0.b-z3.b}, z4.b[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector grouping

usvdot za.s[w8, 0, vgx2], {z0.b-z1.b}, z14.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: usvdot za.s[w8, 0, vgx2], {z0.b-z1.b}, z14.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid lane index

usvdot za.s[w8, 0, vgx4], {z0.b-z3.b}, z14.b[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3]
// CHECK-NEXT: usvdot za.s[w8, 0, vgx4], {z0.b-z3.b}, z14.b[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
