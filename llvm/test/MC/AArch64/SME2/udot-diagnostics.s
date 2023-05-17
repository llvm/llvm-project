// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid select register

udot za.s[w7, 0, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: udot za.s[w7, 0, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot za.s[w12, 0, vgx2], {z0.h-z1.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: udot za.s[w12, 0, vgx2], {z0.h-z1.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid select offset

udot za.s[w8, 8], {z0.h-z1.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: udot za.s[w8, 8], {z0.h-z1.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot za.s[w8, -1], {z0.h-z1.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: udot za.s[w8, -1], {z0.h-z1.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Out of range element index

udot za.s[w8, 0], {z0.h-z1.h}, z0.h[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: udot za.s[w8, 0], {z0.h-z1.h}, z0.h[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot za.s[w8, 0], {z0.h-z3.h}, z0.h[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: udot za.s[w8, 0], {z0.h-z3.h}, z0.h[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// ZPR range constraint

udot za.s[w8, 5], {z0.h-z1.h}, z16.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: udot za.s[w8, 5], {z0.h-z1.h}, z16.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot za.s[w8, 5], {z0.h-z3.h}, z16.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: udot za.s[w8, 5], {z0.h-z3.h}, z16.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
