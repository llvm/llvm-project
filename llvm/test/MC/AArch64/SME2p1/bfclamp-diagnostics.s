// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+b16b16 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

bfclamp {z0.h-z2.h}, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfclamp {z0.h-z2.h}, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfclamp {z23.h-z24.h}, z13.h, z8.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: bfclamp {z23.h-z24.h}, z13.h, z8.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfclamp {z21.h-z24.h}, z10.h, z21.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: bfclamp {z21.h-z24.h}, z10.h, z21.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid Register Suffix

bfclamp {z0.s-z1.s}, z0.h, z4.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfclamp {z0.s-z1.s}, z0.h, z4.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfclamp {z0.h-z3.h}, z5.d, z6.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfclamp {z0.h-z3.h}, z5.d, z6.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
