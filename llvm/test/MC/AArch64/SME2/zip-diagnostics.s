// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

zip {z0.q-z2.q}, z0.q, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: zip {z0.q-z2.q}, z0.q, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zip {z21.h-z22.h}, z10.h, z21.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: zip {z21.h-z22.h}, z10.h, z21.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zip {z0.s-z4.s}, {z0.s-z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: zip {z0.s-z4.s}, {z0.s-z3.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zip {z20.b-z23.b}, {z9.b-z12.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: zip {z20.b-z23.b}, {z9.b-z12.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zip {z1.q-z2.q}, z0.q, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: zip {z1.q-z2.q}, z0.q, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zip {z1.q-z4.q}, z0.q, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: zip {z1.q-z4.q}, z0.q, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
