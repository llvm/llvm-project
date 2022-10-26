// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector select register

fvdot za.s[w7, 0, vgx2], {z0.h-z1.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fvdot za.s[w7, 0, vgx2], {z0.h-z1.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.s[w12, 0, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fvdot za.s[w12, 0, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select offset

fvdot za.s[w8, -1, vgx2], {z0.h-z1.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fvdot za.s[w8, -1, vgx2], {z0.h-z1.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.s[w8, 8, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fvdot za.s[w8, 8, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

fvdot za.s[w8, 0, vgx2], {z0.h-z2.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fvdot za.s[w8, 0, vgx2], {z0.h-z2.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.s[w8, 0, vgx2], {z1.h-z2.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element type
// CHECK-NEXT: fvdot za.s[w8, 0, vgx2], {z1.h-z2.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Matrix Operand

fvdot za.h[w8, 0, vgx2], {z0.h-z2.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fvdot za.h[w8, 0, vgx2], {z0.h-z2.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector grouping

fvdot za.s[w8, 0, vgx4], {z0.h-z1.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fvdot za.s[w8, 0, vgx4], {z0.h-z1.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid lane index

fvdot za.s[w8, 0, vgx2], {z0.h-z1.h}, z0.h[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3]
// CHECK-NEXT: fvdot za.s[w8, 0, vgx2], {z0.h-z1.h}, z0.h[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.s[w8, 0, vgx2], {z0.h-z1.h}, z0.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3]
// CHECK-NEXT: fvdot za.s[w8, 0, vgx2], {z0.h-z1.h}, z0.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
