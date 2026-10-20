// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-b16b16 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

bfmla za.h[w11, 2, vgx2], {z12.h-z14.h}, z8.h[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfmla za.h[w11, 2, vgx2], {z12.h-z14.h}, z8.h[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmla za.h[w11, 2, vgx4], {z12.h-z17.h}, z7.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: bfmla za.h[w11, 2, vgx4], {z12.h-z17.h}, z7.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmla za.h[w10, 3, vgx2], {z10.h-z11.h}, {z21.h-z22.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: bfmla za.h[w10, 3, vgx2], {z10.h-z11.h}, {z21.h-z22.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmla za.h[w11, 7, vgx4], {z12.h-z15.h}, {z9.h-z12.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: bfmla za.h[w11, 7, vgx4], {z12.h-z15.h}, {z9.h-z12.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid indexed-vector or single-vector register

bfmla za.h[w8, 0], {z0.h-z1.h}, z16.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: bfmla za.h[w8, 0], {z0.h-z1.h}, z16.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmla za.h[w8, 1], {z0.h-z3.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: bfmla za.h[w8, 1], {z0.h-z3.h}, z16.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

bfmla za.h[w7, 7, vgx4], {z12.h-z15.h}, {z8.h-z11.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: bfmla za.h[w7, 7, vgx4], {z12.h-z15.h}, {z8.h-z11.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmla za.h[w12, 7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: bfmla za.h[w12, 7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select offset

bfmla za.h[w8, -1, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: bfmla za.h[w8, -1, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmla za.h[w8, 8, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: bfmla za.h[w8, 8, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

bfmla za.d[w8, 7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .h
// CHECK-NEXT: bfmla za.d[w8, 7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector lane index

bfmla za.h[w11, 6, vgx2], {z12.h-z13.h}, z8.h[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: bfmla za.h[w11, 6, vgx2], {z12.h-z13.h}, z8.h[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmla za.h[w11, 6, vgx2], {z12.h-z13.h}, z8.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: bfmla za.h[w11, 6, vgx2], {z12.h-z13.h}, z8.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmla za.h[w11, 7, vgx4], {z12.h-z15.h}, z8.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: bfmla za.h[w11, 7, vgx4], {z12.h-z15.h}, z8.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmla za.h[w11, 7, vgx4], {z12.h-z15.h}, z8.h[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: bfmla za.h[w11, 7, vgx4], {z12.h-z15.h}, z8.h[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
