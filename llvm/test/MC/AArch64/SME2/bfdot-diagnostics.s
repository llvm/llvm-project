// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

bfdot za.s[w8, 0, vgx2], {z0.h-z2.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfdot za.s[w8, 0, vgx2], {z0.h-z2.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot za.s[w8, 0, vgx4], {z1.h-z5.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: bfdot za.s[w8, 0, vgx4], {z1.h-z5.h}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot za.s[w8, 0, vgx2], {z0.h-z1.h}, {z3.h-z4.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: bfdot za.s[w8, 0, vgx2], {z0.h-z1.h}, {z3.h-z4.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid single vector register

bfdot za.s[w8, 0, vgx4], {z0.h-z3.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: bfdot za.s[w8, 0, vgx4], {z0.h-z3.h}, z16.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

bfdot za.s[w7, 0, vgx2], {z0.h-z1.h}, {z3.h-z4.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: bfdot za.s[w7, 0, vgx2], {z0.h-z1.h}, {z3.h-z4.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot za.s[w12, 0, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: bfdot za.s[w12, 0, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select offset

bfdot za.s[w8, -1, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: bfdot za.s[w8, -1, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot za.s[w8, 8, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: bfdot za.s[w8, 8, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot za.s[w8, -1, vgx2], {z0.h-z1.h}, {z3.h-z4.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: bfdot za.s[w8, -1, vgx2], {z0.h-z1.h}, {z3.h-z4.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

bfdot za.h[w8, 0, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: bfdot za.h[w8, 0, vgx4], {z0.h-z3.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector lane index

bfdot za.s[w8, 0, vgx4], {z0.h-z3.h}, z0.h[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: bfdot za.s[w8, 0, vgx4], {z0.h-z3.h}, z0.h[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot za.s[w8, 0, vgx4], {z0.h-z3.h}, z0.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: bfdot za.s[w8, 0, vgx4], {z0.h-z3.h}, z0.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
