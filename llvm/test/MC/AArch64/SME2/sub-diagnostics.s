// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Out of range index offset

sub za.s[w8, 8], {z20.s-z21.s}, z10.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: sub za.s[w8, 8], {z20.s-z21.s}, z10.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub za.d[w8, -1, vgx4], {z0.s-z3.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: sub za.d[w8, -1, vgx4], {z0.s-z3.s}, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

sub za.d[w7, 0], {z0.d-z3.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: sub za.d[w7, 0], {z0.d-z3.d}, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub za.s[w12, 0], {z0.s-z1.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: sub za.s[w12, 0], {z0.s-z1.s}, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Matrix Operand

sub za.h[w8, #0], {z0.h-z3.h}, z4.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .d
// CHECK-NEXT: sub za.h[w8, #0], {z0.h-z3.h}, z4.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector grouping

sub za.s[w8, 0, vgx4], {z0.s-z1.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: za.s[w8, 0, vgx4], {z0.s-z1.s}, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub za.d[w8, 0, vgx2], {z0.d-z3.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: za.d[w8, 0, vgx2], {z0.d-z3.d}, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

