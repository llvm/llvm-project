// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Out of range index offset

add za.s[w8, 8], {z20.s-z21.s}, z10.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: add za.s[w8, 8], {z20.s-z21.s}, z10.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add za.d[w8, -1, vgx4], {z0.s-z3.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: add za.d[w8, -1, vgx4], {z0.s-z3.s}, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     za.s[w8, 8], {z0.s - z1.s}, {z2.s - z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: add     za.s[w8, 8], {z0.s - z1.s}, {z2.s - z3.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     za.s[w8, -1], {z0.s - z3.s}, {z4.s - z7.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: add     za.s[w8, -1], {z0.s - z3.s}, {z4.s - z7.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add za.d[w8, 8, vgx4], {z0.s-z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: za.d[w8, 8, vgx4], {z0.s-z3.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add za.d[w8, -1, vgx4], {z0.s-z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: za.d[w8, -1, vgx4], {z0.s-z3.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector select register

add za.d[w7, 0], {z0.d-z3.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: add za.d[w7, 0], {z0.d-z3.d}, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add za.s[w12, 0], {z0.s-z1.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: add za.s[w12, 0], {z0.s-z1.s}, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     za.s[w7, 0], {z0.s - z1.s}, {z2.s - z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: add     za.s[w7, 0], {z0.s - z1.s}, {z2.s - z3.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:$

// --------------------------------------------------------------------------//
// Invalid Matrix Operand

add za.h[w8, #0], {z0.h-z3.h}, z4.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .d
// CHECK-NEXT: add za.h[w8, #0], {z0.h-z3.h}, z4.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add za.h[w8, 0, vgx2], {z0.s, z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .d
// CHECK-NEXT: za.h[w8, 0, vgx2], {z0.s, z1.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector grouping

add za.s[w8, 0, vgx4], {z0.s-z1.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: za.s[w8, 0, vgx4], {z0.s-z1.s}, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add za.d[w8, 0, vgx2], {z0.d-z3.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: za.d[w8, 0, vgx2], {z0.d-z3.d}, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list.

add za.d[w8, 0], {z0.d,z2.d}, {z0.d,z2.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: add za.d[w8, 0], {z0.d,z2.d}, {z0.d,z2.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add za.s[w10, 3, vgx2], {z10.s-z11.s}, {z21.s-z22.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: add za.s[w10, 3, vgx2], {z10.s-z11.s}, {z21.s-z22.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add za.d[w11, 7, vgx4], {z12.d-z15.d}, {z9.d-z12.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: add za.d[w11, 7, vgx4], {z12.d-z15.d}, {z9.d-z12.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add za.s[w10, 3], {z10.b-z11.b}, {z20.b-z21.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: add za.s[w10, 3], {z10.b-z11.b}, {z20.b-z21.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     za.d[w11, 7], {z28.h - z31.h}, {z28.h - z31.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: add     za.d[w11, 7], {z28.h - z31.h}, {z28.h - z31.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// The tied operands must match, even for vector groups.

add {z0.s-z1.s}, {z2.s-z3.s}, z15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register list
// CHECK-NEXT: add {z0.s-z1.s}, {z2.s-z3.s}, z15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add {z0.s,z1.s}, {z2.s,z3.s}, z15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register list
// CHECK-NEXT: add {z0.s,z1.s}, {z2.s,z3.s}, z15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add {z0.s,z1.s}, {z0.s,z2.s}, z15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: add {z0.s,z1.s}, {z0.s,z2.s}, z15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add {z0.s,z1.s}, {z0.s,z1.s,z2.s}, z15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: add {z0.s,z1.s}, {z0.s,z1.s,z2.s}, z15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add {z0.s,z1.s}, {z0.d,z1.d}, z15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: add {z0.s,z1.s}, {z0.d,z1.d}, z15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

