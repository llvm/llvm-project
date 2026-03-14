// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

bfmlal za.s[w11, 6:7, vgx2], {z12.h-z14.h}, z8.h[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfmlal za.s[w11, 6:7, vgx2], {z12.h-z14.h}, z8.h[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmlal za.s[w11, 6:7, vgx4], {z12.h-z17.h}, z8.h[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: bfmlal za.s[w11, 6:7, vgx4], {z12.h-z17.h}, z8.h[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmlal za.s[w10, 2:3, vgx2], {z10.h-z11.h}, {z21.h-z22.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: bfmlal za.s[w10, 2:3, vgx2], {z10.h-z11.h}, {z21.h-z22.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid indexed-vector register

bfmlal za.s[w8, 0:1], z0.h, z17.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: bfmlal za.s[w8, 0:1], z0.h, z17.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmlal za.s[w8, 0:1], z0.h, z30.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: bfmlal za.s[w8, 0:1], z0.h, z30.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

bfmlal za.s[w7, 6:7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: bfmlal za.s[w7, 6:7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmlal za.s[w12, 6:7, vgx4], {z12.h-z15.h}, {z8.h-z11.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: bfmlal za.s[w12, 6:7, vgx4], {z12.h-z15.h}, {z8.h-z11.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select offset

bfmlal za.s[w8, 6:9, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfmlal za.s[w8, 6:9, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmlal za.s[w8, 9:10, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 2 in the range [0, 6] or [0, 14] depending on the instruction, and the second immediate is immf + 1.
// CHECK-NEXT: bfmlal za.s[w8, 9:10, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

bfmlal za.h[w8, 6:7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: bfmlal za.h[w8, 6:7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector lane index

bfmlal za.s[w11, 6:7, vgx4], {z12.h-z15.h}, z8.h[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: bfmlal za.s[w11, 6:7, vgx4], {z12.h-z15.h}, z8.h[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmlal za.s[w11, 6:7, vgx4], {z12.h-z15.h}, z8.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: bfmlal za.s[w11, 6:7, vgx4], {z12.h-z15.h}, z8.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
