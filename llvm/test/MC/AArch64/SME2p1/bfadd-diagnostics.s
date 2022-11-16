// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+b16b16 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Out of range index offset

bfadd za.h[w8, 8], {z20.h-z21.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: bfadd za.h[w8, 8], {z20.h-z21.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfadd za.h[w8, -1, vgx4], {z0.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: bfadd za.h[w8, -1, vgx4], {z0.h-z3.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

bfadd za.h[w7, 0], {z20.h-z21.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: bfadd za.h[w7, 0], {z20.h-z21.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfadd za.h[w12, 0, vgx4], {z20.h-z23.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: bfadd za.h[w12, 0, vgx4], {z20.h-z23.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

bfadd za.h[w8, 3], {z20.h-z22.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfadd za.h[w8, 3], {z20.h-z22.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfadd za.h[w8, 3, vgx4], {z21.h-z24.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: bfadd za.h[w8, 3, vgx4], {z21.h-z24.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid suffixes

bfadd za.h[w8, 3, vgx4], {z20.s-z23.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfadd za.h[w8, 3, vgx4], {z20.s-z23.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfadd za.d[w8, 3, vgx4], {z20.h-z23.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .h
// CHECK-NEXT: bfadd za.d[w8, 3, vgx4], {z20.h-z23.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
