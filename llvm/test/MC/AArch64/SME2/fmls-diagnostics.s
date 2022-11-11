// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-f64f64 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Out of range index offset

fmls za.s[w8, 8], {z20.s-z21.s}, z10.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fmls za.s[w8, 8], {z20.s-z21.s}, z10.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmls za.d[w8, -1, vgx4], {z0.s-z3.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fmls za.d[w8, -1, vgx4], {z0.s-z3.s}, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

fmls za.d[w7, 0], {z0.d-z3.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fmls za.d[w7, 0], {z0.d-z3.d}, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmls za.s[w12, 0], {z0.s-z1.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fmls za.s[w12, 0], {z0.s-z1.s}, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Matrix Operand

fmls za.h[w8, #0], {z0.h-z3.h}, z4.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .d
// CHECK-NEXT: fmls za.h[w8, #0], {z0.h-z3.h}, z4.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector grouping

fmls za.s[w8, 0, vgx4], {z0.s-z1.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: za.s[w8, 0, vgx4], {z0.s-z1.s}, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmls za.d[w8, 0, vgx2], {z0.d-z3.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: za.d[w8, 0, vgx2], {z0.d-z3.d}, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list.

fmls za.d[w8, 0], {z0.d,z2.d}, {z0.d,z2.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmls za.d[w8, 0], {z0.d,z2.d}, {z0.d,z2.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmls za.s[w10, 3, vgx2], {z10.s-z11.s}, {z21.s-z22.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: fmls za.s[w10, 3, vgx2], {z10.s-z11.s}, {z21.s-z22.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmls za.d[w11, 7, vgx4], {z12.d-z15.d}, {z9.d-z12.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: fmls za.d[w11, 7, vgx4], {z12.d-z15.d}, {z9.d-z12.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmls za.s[w10, 3], {z10.b-z11.b}, {z20.b-z21.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmls za.s[w10, 3], {z10.b-z11.b}, {z20.b-z21.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmls     za.d[w11, 7], {z28.h - z31.h}, {z28.h - z31.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmls     za.d[w11, 7], {z28.h - z31.h}, {z28.h - z31.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
