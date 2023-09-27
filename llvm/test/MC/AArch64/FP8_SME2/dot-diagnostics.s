// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f8f16,+sme-f8f32  2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector select register

fdot    za.h[w8, 0, vgx2], {z0.h-z1.h}, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot    za.h[w8, 0, vgx2], {z0.h-z1.h}, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.h[w11, 7], {z31.b-z2.b}, z15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: fdot    za.h[w11, 7], {z31.b-z2.b}, z15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.h[w11, 7, vgx2], {z28.b-z31.b}, {z0.b-z3.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot    za.h[w11, 7, vgx2], {z28.b-z31.b}, {z0.b-z3.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.s[w11, 7], {z29.b-z30.b}, {z30.b-z31.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot    za.s[w11, 7], {z29.b-z30.b}, {z30.b-z31.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.h[w11, 7], {z30.b-z0.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot    za.h[w11, 7], {z30.b-z0.b}, z15.
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector select offset

fvdott  za.s[w11, -1, vgx4], {z30.b-z31.b}, z15.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fvdott  za.s[w11, -1, vgx4], {z30.b-z31.b}, z15.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdott  za.s[w8, -1, vgx4], {z0.b-z1.b}, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fvdott  za.s[w8, -1, vgx4], {z0.b-z1.b}, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot   za.h[w11, -1], {z30.b-z31.b}, z15.b[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fvdot   za.h[w11, -1], {z30.b-z31.b}, z15.b[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.s[w11, -1], {z28.b-z31.b}, z15.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fdot    za.s[w11, -1], {z28.b-z31.b}, z15.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdott  za.s[w11, 8, vgx4], {z30.b-z31.b}, z15.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fvdott  za.s[w11, 8, vgx4], {z30.b-z31.b}, z15.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdott  za.s[w8, 8, vgx4], {z0.b-z1.b}, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fvdott  za.s[w8, 8, vgx4], {z0.b-z1.b}, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot   za.h[w11, 8], {z30.b-z31.b}, z15.b[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fvdot   za.h[w11, 8], {z30.b-z31.b}, z15.b[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.s[w11, 8], {z28.b-z31.b}, z15.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fdot    za.s[w11, 8], {z28.b-z31.b}, z15.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

fdot    za.s[w11, 7, vgx4], {z29.b-z1.b}, {z29.b-z1.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: fdot    za.s[w11, 7, vgx4], {z29.b-z1.b}, {z29.b-z1.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.h[w11, 7], {z30.b-z2.b}, {z0.b-z3.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: fdot    za.h[w11, 7], {z30.b-z2.b}, {z0.b-z3.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.s[w8, 0], {z31.b-z3.b}, {z31.b-z3.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: fdot    za.s[w8, 0], {z31.b-z3.b}, {z31.b-z3.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.s[w11, 7, vgx2], {z30.b-z31.b}, {z0.b-z4.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: fdot    za.s[w11, 7, vgx2], {z30.b-z31.b}, {z0.b-z4.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix
fdot    za.d[w11, 7, vgx4], {z31.b-z2.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fdot    za.d[w11, 7, vgx4], {z31.b-z2.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za[w11, 7], {z28.b-z31.b}, {z28.b-z31.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fdot    za[w11, 7], {z28.b-z31.b}, {z28.b-z31.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.b[w11, 7], {z31.b-z0.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fdot    za.b[w11, 7], {z31.b-z0.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.b[w11, 7, vgx2], {z30.h-z31.h}, {z30.h-z31.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fdot    za.b[w11, 7, vgx2], {z30.h-z31.h}, {z30.h-z31.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za[w11, 7, vgx4], {z31.b-z2.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fdot    za[w11, 7, vgx4], {z31.b-z2.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.d[w11, 7], {z28.b-z31.b}, {z28.b-z31.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fdot    za.d[w11, 7], {z28.b-z31.b}, {z28.b-z31.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

fdot    za.h[w7, 7, vgx4], {z31.b-z2.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fdot    za.h[w7, 7, vgx4], {z31.b-z2.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.h[w, 0, vgx2], {z0.b-z1.b}, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fdot    za.h[w, 0, vgx2], {z0.b-z1.b}, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.s[w12, 0], {z0.b-z3.b}, {z0.b-z3.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fdot    za.s[w12, 0], {z0.b-z3.b}, {z0.b-z3.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid indexed-vector or single-vector register

fdot za.h[w8, 0], {z0.b-z1.b}, z16.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: fdot za.h[w8, 0], {z0.b-z1.b}, z16.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot za.s[w8, 0], {z0.b-z1.b}, z16.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT:  fdot za.s[w8, 0], {z0.b-z1.b}, z16.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector grouping

fdot    za.h[w11, 7], {z28.b-z31.b}, {z0.b-z2.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot    za.h[w11, 7], {z28.b-z31.b}, {z0.b-z2.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.h[w11, 7, vgx4], {z31.b-z0.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot    za.h[w11, 7, vgx4], {z31.b-z0.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid lane index

fdot    za.h[w8, 0, vgx2], {z0.b-z1.b}, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: fdot    za.h[w8, 0, vgx2], {z0.b-z1.b}, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.h[w11, 7], {z30.b-z31.b}, z15.b[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: fdot    za.h[w11, 7], {z30.b-z31.b}, z15.b[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.s[w8, 0], {z0.b-z1.b}, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fdot    za.s[w8, 0], {z0.b-z1.b}, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot    za.s[w11, 7], {z30.b-z31.b}, z15.b[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fdot    za.s[w11, 7], {z30.b-z31.b}, z15.b[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot   za.h[w8, 0, vgx2], {z0.b-z1.b}, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: fvdot   za.h[w8, 0, vgx2], {z0.b-z1.b}, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot   za.h[w11, 7], {z30.b-z31.b}, z15.b[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: fvdot   za.h[w11, 7], {z30.b-z31.b}, z15.b[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdotb  za.s[w8, 0, vgx4], {z0.b-z1.b}, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fvdotb  za.s[w8, 0, vgx4], {z0.b-z1.b}, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdott  za.s[w11, 7, vgx4], {z30.b-z31.b}, z15.b[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fvdott  za.s[w11, 7, vgx4], {z30.b-z31.b}, z15.b[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
