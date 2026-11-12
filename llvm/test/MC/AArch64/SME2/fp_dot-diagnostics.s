// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f16f8f16 2>&1 < %s \
// RUN:   | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector select register

fdot za.h[w7, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fdot za.h[w7, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.h[w7, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fvdot za.h[w7, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.h[w12, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fvdot za.h[w12, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot za.h[w12, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fdot za.h[w12, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select offset

fdot za.h[w8, -1, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fdot za.h[w8, -1, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.h[w8, -1, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fvdot za.h[w8, -1, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.h[w8, 8, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fvdot za.h[w8, 8, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot za.h[w8, 8, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: fdot za.h[w8, 8, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid matrix operand

fdot za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fvdot za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector grouping

fvdot za.h[w8, 0, vgx4], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fvdot za.h[w8, 0, vgx4], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot za.h[w8, 0, vgx4], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot za.h[w8, 0, vgx4], { z0.b, z1.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

fdot za.h[w8, 0, vgx2], { z0.b, z2.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot za.h[w8, 0, vgx2], { z0.b, z2.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.h[w8, 0, vgx2], { z0.b, z2.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fvdot za.h[w8, 0, vgx2], { z0.b, z2.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.h[w8, 0, vgx2], { z1.b, z2.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: fvdot za.h[w8, 0, vgx2], { z1.b, z2.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot za.h[w8, 0, vgx2], { z1.b, z2.b }, z0.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot za.h[w8, 0, vgx2], { z1.b, z2.b }, z0.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid indexed vector register

fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z16.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z16.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z16.h[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z16.h[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.s[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.s[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.s[0], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.s[0], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid lane index

fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[-1], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[-1], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[-1], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[-1], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[4], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[4], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[4], z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[4], z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Zk register

fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z19.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20.b..z23.b or z28.b..z31.b
// CHECK-NEXT: fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z19.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z19.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20.b..z23.b or z28.b..z31.b
// CHECK-NEXT: fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z19.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z24.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20.b..z23.b or z28.b..z31.b
// CHECK-NEXT: fvdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z24.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z24.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20.b..z23.b or z28.b..z31.b
// CHECK-NEXT: fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.h[0], z24.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
