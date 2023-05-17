// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

fclamp {z0.h-z2.h}, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fclamp {z0.h-z2.h}, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fclamp {z0.d-z4.d}, z5.d, z6.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: fclamp {z0.d-z4.d}, z5.d, z6.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fclamp {z23.s-z24.s}, z13.s, z8.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element type
// CHECK-NEXT: fclamp {z23.s-z24.s}, z13.s, z8.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fclamp {z21.h-z24.h}, z10.h, z21.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element type
// CHECK-NEXT: fclamp {z21.h-z24.h}, z10.h, z21.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid Register Suffix

fclamp {z0.h-z1.h}, z0.h, z4.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fclamp {z0.h-z1.h}, z0.h, z4.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fclamp {z0.s-z3.s}, z5.d, z6.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fclamp {z0.s-z3.s}, z5.d, z6.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
