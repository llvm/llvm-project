// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

sqdmulh {z0.h-z2.h}, {z0.h-z1.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqdmulh {z0.h-z2.h}, {z0.h-z1.h}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z28.s-z31.s}, {z0.s-z4.s}, z15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: sqdmulh {z28.s-z31.s}, {z0.s-z4.s}, z15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z1.d-z4.d}, {z0.d-z3.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: sqdmulh {z1.d-z4.d}, {z0.d-z3.d}, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z28.b-z29.b}, {z1.b-z2.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: sqdmulh {z28.b-z29.b}, {z1.b-z2.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z28.h-z29.h}, {z1.h-z2.h}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: sqdmulh {z28.h-z29.h}, {z1.h-z2.h}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z1.d-z4.d}, {z1.d-z4.d}, {z8.d-z11.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: sqdmulh {z1.d-z4.d}, {z1.d-z4.d}, {z8.d-z11.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z1.d-z2.d}, {z1.d-z2.d}, {z2.d-z3.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: sqdmulh {z1.d-z2.d}, {z1.d-z2.d}, {z2.d-z3.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

// --------------------------------------------------------------------------//
// Invalid single vector register

sqdmulh {z28.b-z29.b}, {z0.b-z1.b}, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: sqdmulh {z28.b-z29.b}, {z0.b-z1.b}, z16.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid register suffix

sqdmulh {z0.d-z3.d}, {z0.d-z3.d}, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.d..z15.d
// CHECK-NEXT: sqdmulh {z0.d-z3.d}, {z0.d-z3.d}, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z0.d-z3.h}, {z0.d-z3.d}, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: sqdmulh {z0.d-z3.h}, {z0.d-z3.d}, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// The tied operands must match, even for vector groups.

sqdmulh {z0.s-z1.s}, {z2.s-z3.s}, z15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register list
// CHECK-NEXT: sqdmulh {z0.s-z1.s}, {z2.s-z3.s}, z15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z0.s,z1.s}, {z2.s,z3.s}, z15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register list
// CHECK-NEXT: sqdmulh {z0.s,z1.s}, {z2.s,z3.s}, z15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z0.s,z1.s}, {z0.s,z2.s}, z15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqdmulh {z0.s,z1.s}, {z0.s,z2.s}, z15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z0.s,z1.s}, {z0.s,z1.s,z2.s}, z15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqdmulh {z0.s,z1.s}, {z0.s,z1.s,z2.s}, z15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z0.s,z1.s}, {z0.d,z1.d}, z15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqdmulh {z0.s,z1.s}, {z0.d,z1.d}, z15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z2.d,z3.d}, {z0.d,z1.d}, {z4.d,z5.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register list
// CHECK-NEXT: sqdmulh {z2.d,z3.d}, {z0.d,z1.d}, {z4.d,z5.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdmulh {z0.d-z3.d}, {z4.d-z7.d}, {z0.d-z3.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register list
// CHECK-NEXT: sqdmulh {z0.d-z3.d}, {z4.d-z7.d}, {z0.d-z3.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
