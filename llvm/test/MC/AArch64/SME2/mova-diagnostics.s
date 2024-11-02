// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Multi-vector sequence constraints

mova {z1.d-z2.d}, za.d[w12]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: mova {z1.d-z2.d}, za.d[w12]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova {z1.d-z4.d}, za.d[w12]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: mova {z1.d-z4.d}, za.d[w12]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid index offset

mova {z0.s, z1.s}, za0h.s[w12, 1:2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 2 in the range [0, 2], and the second immediate is immf + 1.
// CHECK-NEXT: mova {z0.s, z1.s}, za0h.s[w12, 1:2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova {z0.s, z1.s}, za0h.s[w12, 3:4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 2 in the range [0, 2], and the second immediate is immf + 1.
// CHECK-NEXT: mova {z0.s, z1.s}, za0h.s[w12, 3:4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova {z0.s, z1.s}, za0h.s[w12, 0:2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mova {z0.s, z1.s}, za0h.s[w12, 0:2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova    {z20.d-z21.d}, za2h.d[w14, 0:3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mova    {z20.d-z21.d}, za2h.d[w14, 0:3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova    {z16.s-z19.s}, za1h.s[w14, 0:1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mova    {z16.s-z19.s}, za1h.s[w14, 0:1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid index (expected range)

mova {z0.b-z3.b}, za0h.b[w13, 0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mova {z0.b-z3.b}, za0h.b[w13, 0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Operands are not consistent

mova za.h[w8, 0], {z0.s-z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mova za.h[w8, 0], {z0.s-z3.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov za.h[w8, 0], {z0.s-z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT:mov za.h[w8, 0], {z0.s-z3.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova {z0.s-z3.s}, za.b[w8, 0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: mova {z0.s-z3.s}, za.b[w8, 0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov {z0.h-z3.h}, za.d[w8, 0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .h
// CHECK-NEXT: mov {z0.h-z3.h}, za.d[w8, 0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
