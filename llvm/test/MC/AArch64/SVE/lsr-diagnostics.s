// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

lsr z30.b, z10.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8]
// CHECK-NEXT: lsr z30.b, z10.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsr z18.b, z27.b, #9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8]
// CHECK-NEXT: lsr z18.b, z27.b, #9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsr z26.h, z4.h, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: lsr z26.h, z4.h, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsr z25.h, z10.h, #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: lsr z25.h, z10.h, #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsr z17.s, z0.s, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32]
// CHECK-NEXT: lsr z17.s, z0.s, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsr z0.s, z15.s, #33
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32]
// CHECK-NEXT: lsr z0.s, z15.s, #33
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsr z4.d, z13.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]
// CHECK-NEXT: lsr z4.d, z13.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsr z26.d, z26.d, #65
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]
// CHECK-NEXT: lsr z26.d, z26.d, #65
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Source and Destination Registers must match
lsr z0.b, p0/m, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: lsr z0.b, p0/m, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Element sizes must match
lsr z0.b, p0/m, z0.b, z1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lsr z0.b, p0/m, z0.b, z1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Predicate not in restricted predicate range
lsr z0.b, p8/m, z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: lsr z0.b, p8/m, z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
