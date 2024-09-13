// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-b16b16 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Out of range index offset

bfsub za.h[w8, 8], {z20.h-z21.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: bfsub za.h[w8, 8], {z20.h-z21.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfsub za.h[w8, -1, vgx4], {z0.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: bfsub za.h[w8, -1, vgx4], {z0.h-z3.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

bfsub za.h[w7, 0], {z20.h-z21.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: bfsub za.h[w7, 0], {z20.h-z21.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfsub za.h[w12, 0, vgx4], {z20.h-z23.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: bfsub za.h[w12, 0, vgx4], {z20.h-z23.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

bfsub za.h[w8, 3], {z20.h-z22.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfsub za.h[w8, 3], {z20.h-z22.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfsub za.h[w8, 3, vgx4], {z21.h-z24.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: bfsub za.h[w8, 3, vgx4], {z21.h-z24.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid suffixes

bfsub za.h[w8, 3, vgx4], {z20.s-z23.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfsub za.h[w8, 3, vgx4], {z20.s-z23.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfsub za.d[w8, 3, vgx4], {z20.h-z23.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .h
// CHECK-NEXT: bfsub za.d[w8, 3, vgx4], {z20.h-z23.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
