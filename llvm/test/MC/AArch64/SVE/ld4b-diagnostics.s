// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Immediate out of lower bound [-32, 28].

ld4b {z12.b, z13.b, z14.b, z15.b}, p4/z, [x12, #-36, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28].
// CHECK-NEXT: ld4b {z12.b, z13.b, z14.b, z15.b}, p4/z, [x12, #-36, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld4b {z7.b, z8.b, z9.b, z10.b}, p3/z, [x1, #32, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28].
// CHECK-NEXT: ld4b {z7.b, z8.b, z9.b, z10.b}, p3/z, [x1, #32, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Immediate not a multiple of four.

ld4b {z12.b, z13.b, z14.b, z15.b}, p4/z, [x12, #-7, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28].
// CHECK-NEXT: ld4b {z12.b, z13.b, z14.b, z15.b}, p4/z, [x12, #-7, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld4b {z7.b, z8.b, z9.b, z10.b}, p3/z, [x1, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28].
// CHECK-NEXT: ld4b {z7.b, z8.b, z9.b, z10.b}, p3/z, [x1, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// error: restricted predicate has range [0, 7].

ld4b {z2.b, z3.b, z4.b, z5.b}, p8/z, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld4b {z2.b, z3.b, z4.b, z5.b}, p8/z, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

ld4b { }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: ld4b { }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld4b { z0.b, z1.b, z2.b, z3.b, z4.b }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: ld4b { z0.b, z1.b, z2.b, z3.b, z4.b }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld4b { z0.b, z1.b, z2.b, z3.h }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: ld4b { z0.b, z1.b, z2.b, z3.h }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld4b { z0.b, z1.b, z3.b, z5.b }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must be sequential
// CHECK-NEXT: ld4b { z0.b, z1.b, z3.b, z5.b }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld4b { v0.16b, v1.16b, v2.16b }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld4b { v0.16b, v1.16b, v2.16b }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
