// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

smlsll za.d[w11, 6:7, vgx2], {z12.h-z14.h}, z8.h[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: smlsll za.d[w11, 6:7, vgx2], {z12.h-z14.h}, z8.h[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smlsll za.d[w11, 6:7, vgx4], {z12.h-z17.h}, z8.h[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: smlsll za.d[w11, 6:7, vgx4], {z12.h-z17.h}, z8.h[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smlsll za.s[w10, 4:7], {z8.b-z11.b}, {z21.b-z24.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: smlsll za.s[w10, 4:7], {z8.b-z11.b}, {z21.b-z24.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid indexed-vector register

smlsll za.s[w10, 0:3], z19.b, z4.s[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: smlsll za.s[w10, 0:3], z19.b, z4.s[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smlsll za.d[w10, 4:7], z10.h, z30.h[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: smlsll za.d[w10, 4:7], z10.h, z30.h[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

smlsll za.s[w7, 6:7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: smlsll za.s[w7, 6:7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smlsll za.s[w12, 6:7, vgx4], {z12.b-z15.b}, z8.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: smlsll za.s[w12, 6:7, vgx4], {z12.b-z15.b}, z8.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select offset

smlsll za.s[w11, 4:8], {z30.b-z31.b}, z15.b[15]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: smlsll za.s[w11, 4:8], {z30.b-z31.b}, z15.b[15]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smlsll za.d[w8, 5:8, vgx2], {z22.h-z23.h}, z14.h[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 4 in the range [0, 4] or [0, 12] depending on the instruction, and the second immediate is immf + 3.
// CHECK-NEXT: smlsll za.d[w8, 5:8, vgx2], {z22.h-z23.h}, z14.h[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

smlsll za.h[w8, 6:7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .d
// CHECK-NEXT: smlsll za.h[w8, 6:7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector lane index

smlsll  za.s[w8, 0:3], {z0.b-z3.b}, z0.b[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: smlsll  za.s[w8, 0:3], {z0.b-z3.b}, z0.b[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smlsll  za.s[w8, 0:3], {z0.b-z3.b}, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: smlsll  za.s[w8, 0:3], {z0.b-z3.b}, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
