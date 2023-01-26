// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid select register

sudot za.s[w7, 0, vgx4], {z0.b-z3.b}, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: sudot za.s[w7, 0, vgx4], {z0.b-z3.b}, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sudot za.s[w12, 0, vgx2], {z0.b-z1.b}, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: sudot za.s[w12, 0, vgx2], {z0.b-z1.b}, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid select offset

sudot za.s[w8, 8], {z0.b-z1.b}, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: sudot za.s[w8, 8], {z0.b-z1.b}, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sudot za.s[w8, -1], {z0.b-z3.b}, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: sudot za.s[w8, -1], {z0.b-z3.b}, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Out of range element index

sudot za.s[w8, 0], {z0.b-z1.b}, z0.b[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: sudot za.s[w8, 0], {z0.b-z1.b}, z0.b[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sudot za.s[w8, 0], {z0.b-z3.b}, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: sudot za.s[w8, 0], {z0.b-z3.b}, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// ZPR range constraint

sudot za.s[w8, 5], {z0.b-z1.b}, z16.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: sudot za.s[w8, 5], {z0.b-z1.b}, z16.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sudot za.s[w8, 5], {z0.b-z3.b}, z16.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: sudot za.s[w8, 5], {z0.b-z3.b}, z16.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
