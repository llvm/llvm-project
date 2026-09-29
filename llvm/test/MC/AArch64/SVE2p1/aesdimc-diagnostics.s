// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve-aes2,+sve2p1 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

aesdimc {z0.b-z2.b}, {z0.b-z2.b}, z0.q[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: aesdimc {z0.b-z2.b}, {z0.b-z2.b}, z0.q[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesdimc {z0.d-z1.d}, {z0.d-z1.d}, z0.q[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: aesdimc {z0.d-z1.d}, {z0.d-z1.d}, z0.q[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesdimc {z0.s-z3.s}, {z0.s-z3.s}, z0.q[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: aesdimc {z0.s-z3.s}, {z0.s-z3.s}, z0.q[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesdimc {z0.b-z0.b}, {z0.b-z0.b}, z0.q[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: aesdimc {z0.b-z0.b}, {z0.b-z0.b}, z0.q[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesdimc {z3.b-z7.b}, {z3.b-z7.b}, z0.q[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: aesdimc {z3.b-z7.b}, {z3.b-z7.b}, z0.q[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesdimc {z3.b-z4.b}, {z3.b-z4.b}, z0.q[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: aesdimc {z3.b-z4.b}, {z3.b-z4.b}, z0.q[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesdimc {z5.b-z8.b}, {z5.b-z8.b}, z0.q[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: aesdimc {z5.b-z8.b}, {z5.b-z8.b}, z0.q[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid second source vector width

aesdimc {z0.b-z1.b}, {z0.b-z1.b}, z0.d[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: aesdimc {z0.b-z1.b}, {z0.b-z1.b}, z0.d[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate index

aesdimc {z0.b-z1.b}, {z0.b-z1.b}, z0.q[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: aesdimc {z0.b-z1.b}, {z0.b-z1.b}, z0.q[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.q[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.q[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Source and Destination Registers must match

aesdimc {z0.b-z1.b}, {z2.b-z3.b}, z0.q[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register list
// CHECK-NEXT: aesdimc {z0.b-z1.b}, {z2.b-z3.b}, z0.q[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesdimc {z0.b-z3.b}, {z4.b-z7.b}, z0.q[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register list
// CHECK-NEXT: aesdimc {z0.b-z3.b}, {z4.b-z7.b}, z0.q[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

aesdimc {z0.b-z3.b}, {z0.h-z3.h}, z0.q[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: aesdimc {z0.b-z3.b}, {z0.h-z3.h}, z0.q[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: