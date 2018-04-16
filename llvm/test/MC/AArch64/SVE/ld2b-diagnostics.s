// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Immediate out of lower bound [-16, 14].

ld2b {z12.b, z13.b}, p4/z, [x12, #-18, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: ld2b {z12.b, z13.b}, p4/z, [x12, #-18, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2b {z7.b, z8.b}, p3/z, [x1, #16, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: ld2b {z7.b, z8.b}, p3/z, [x1, #16, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Immediate not a multiple of two.

ld2b {z12.b, z13.b}, p4/z, [x12, #-7, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: ld2b {z12.b, z13.b}, p4/z, [x12, #-7, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2b {z7.b, z8.b}, p3/z, [x1, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: ld2b {z7.b, z8.b}, p3/z, [x1, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// error: restricted predicate has range [0, 7].

ld2b {z2.b, z3.b}, p8/z, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld2b {z2.b, z3.b}, p8/z, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

ld2b { }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: ld2b { }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2b { z0.b, z1.b, z2.b }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld2b { z0.b, z1.b, z2.b }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2b { z0.b, z1.h }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: ld2b { z0.b, z1.h }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2b { z0.b, z2.b }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must be sequential
// CHECK-NEXT: ld2b { z0.b, z2.b }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2b { v0.2d, v1.2d }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld2b { v0.2d, v1.2d }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
