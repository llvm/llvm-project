// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

urshl {z0.h-z2.h}, {z0.h-z1.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: urshl {z0.h-z2.h}, {z0.h-z1.h}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshl {z0.s-z1.s}, {z2.s-z4.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: urshl {z0.s-z1.s}, {z2.s-z4.s}, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshl {z20.d-z23.d}, {z20.d-z23.d}, {z8.d-z12.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: urshl {z20.d-z23.d}, {z20.d-z23.d}, {z8.d-z12.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshl {z29.b-z30.b}, {z30.b-z31.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: urshl {z29.b-z30.b}, {z30.b-z31.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshl {z20.h-z23.h}, {z21.h-z24.h}, {z8.h-z11.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: urshl {z20.h-z23.h}, {z21.h-z24.h}, {z8.h-z11.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

urshl {z28.b-z31.b}, {z28.b-z31.b}, {z27.b-z30.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: urshl {z28.b-z31.b}, {z28.b-z31.b}, {z27.b-z30.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Single Register

urshl {z20.h-z21.h}, {z20.h-z21.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: urshl {z20.h-z21.h}, {z20.h-z21.h}, z16.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

urshl {z0.d-z3.d}, {z0.d-z3.d}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.d..z15.d
// CHECK-NEXT: urshl {z0.d-z3.d}, {z0.d-z3.d}, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
