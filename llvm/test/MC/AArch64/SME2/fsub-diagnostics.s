// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-f64f64 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Out of range index offset

fsub za.d[w8, 8, vgx2], {z0.d, z1.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fsub za.d[w8, 8, vgx2], {z0.d, z1.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsub za.s[w8, -1, vgx4], {z0.s-z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fsub za.s[w8, -1, vgx4], {z0.s-z3.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

fsub za.d[w7, 7, vgx4], {z0.d-z3.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fsub za.d[w7, 7, vgx4], {z0.d-z3.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsub za.s[w12, 7, vgx2], {z0.s, z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fsub za.s[w12, 7, vgx2], {z0.s, z1.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

fsub za.d[w8, 0, vgx4], {z0.d-z4.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: fsub za.d[w8, 0, vgx4], {z0.d-z4.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsub za.s[w8, 0, vgx2], {z1.s-z2.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: fsub za.s[w8, 0, vgx2], {z1.s-z2.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsub za.s[w8, 0, vgx4], {z1.s-z4.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: fsub za.s[w8, 0, vgx4], {z1.s-z4.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

