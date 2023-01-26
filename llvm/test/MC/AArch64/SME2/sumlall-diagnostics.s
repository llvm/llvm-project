// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

sumlall za.s[w11, 4:7, vgx2], {z12.h-z14.h}, z8.h[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sumlall za.s[w11, 4:7, vgx2], {z12.h-z14.h}, z8.h[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sumlall za.s[w11, 4:7, vgx4], {z12.h-z17.h}, z8.h[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: sumlall za.s[w11, 4:7, vgx4], {z12.h-z17.h}, z8.h[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid indexed-vector register

sumlall za.s[w10, 0:3], z19.b, z4.s[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: sumlall za.s[w10, 0:3], z19.b, z4.s[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sumlall za.s[w10, 4:7], z10.b, z30.b[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: sumlall za.s[w10, 4:7], z10.b, z30.b[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

sumlall za.s[w7, 6:7, vgx2], {z12.b-z13.b}, z8.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: sumlall za.s[w7, 6:7, vgx2], {z12.b-z13.b}, z8.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sumlall za.s[w12, 6:7, vgx2], {z12.b-z13.b}, z8.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: sumlall za.s[w12, 6:7, vgx2], {z12.b-z13.b}, z8.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select offset

sumlall za.s[w11, 4:8], {z30.b-z31.b}, z15.b[15]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sumlall za.s[w11, 4:8], {z30.b-z31.b}, z15.b[15]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sumlall za.s[w8, 5:8, vgx2], {z22.b-z23.b}, z14.b[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 4 in the range [0, 4] or [0, 12] depending on the instruction, and the second immediate is immf + 3.
// CHECK-NEXT: sumlall za.s[w8, 5:8, vgx2], {z22.b-z23.b}, z14.b[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

sumlall za.h[w8, 6:7, vgx2], {z12.b-z13.b}, z8.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: sumlall za.h[w8, 6:7, vgx2], {z12.b-z13.b}, z8.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector lane index

sumlall  za.s[w8, 0:3], {z0.b-z1.b}, z0.b[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: sumlall  za.s[w8, 0:3], {z0.b-z1.b}, z0.b[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sumlall  za.s[w8, 0:3], {z0.b-z1.b}, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: sumlall  za.s[w8, 0:3], {z0.b-z1.b}, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
