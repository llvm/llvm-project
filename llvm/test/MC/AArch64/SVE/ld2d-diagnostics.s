// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Immediate out of lower bound [-16, 14].

ld2d {z12.d, z13.d}, p4/z, [x12, #-18, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: ld2d {z12.d, z13.d}, p4/z, [x12, #-18, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2d {z7.d, z8.d}, p3/z, [x1, #16, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: ld2d {z7.d, z8.d}, p3/z, [x1, #16, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Immediate not a multiple of two.

ld2d {z12.d, z13.d}, p4/z, [x12, #-7, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: ld2d {z12.d, z13.d}, p4/z, [x12, #-7, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2d {z7.d, z8.d}, p3/z, [x1, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: ld2d {z7.d, z8.d}, p3/z, [x1, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

ld2d { z0.d, z1.d }, p0/z, [x0, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: ld2d { z0.d, z1.d }, p0/z, [x0, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2d { z0.d, z1.d }, p0/z, [x0, xzr]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: ld2d { z0.d, z1.d }, p0/z, [x0, xzr]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2d { z0.d, z1.d }, p0/z, [x0, x0, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: ld2d { z0.d, z1.d }, p0/z, [x0, x0, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2d { z0.d, z1.d }, p0/z, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: ld2d { z0.d, z1.d }, p0/z, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2d { z0.d, z1.d }, p0/z, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: ld2d { z0.d, z1.d }, p0/z, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// error: restricted predicate has range [0, 7].

ld2d {z2.d, z3.d}, p8/z, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld2d {z2.d, z3.d}, p8/z, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

ld2d { }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: ld2d { }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2d { z0.d, z1.d, z2.d }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld2d { z0.d, z1.d, z2.d }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2d { z0.d, z1.b }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: ld2d { z0.d, z1.b }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2d { z0.d, z2.d }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must be sequential
// CHECK-NEXT: ld2d { z0.d, z2.d }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld2d { v0.2d, v1.2d }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld2d { v0.2d, v1.2d }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z21.d, p5/z, z28.d
ld2d    { z21.d, z22.d }, p5/z, [x10, #10, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ld2d    { z21.d, z22.d }, p5/z, [x10, #10, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z21, z28
ld2d    { z21.d, z22.d }, p5/z, [x10, #10, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ld2d    { z21.d, z22.d }, p5/z, [x10, #10, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
