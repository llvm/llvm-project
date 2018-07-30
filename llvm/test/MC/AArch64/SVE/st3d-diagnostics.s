// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Immediate out of lower bound [-24, 21].

st3d {z12.d, z13.d, z14.d}, p4, [x12, #-27, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: st3d {z12.d, z13.d, z14.d}, p4, [x12, #-27, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3d {z7.d, z8.d, z9.d}, p3, [x1, #24, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: st3d {z7.d, z8.d, z9.d}, p3, [x1, #24, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Immediate not a multiple of three.

st3d {z12.d, z13.d, z14.d}, p4, [x12, #-7, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: st3d {z12.d, z13.d, z14.d}, p4, [x12, #-7, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3d {z7.d, z8.d, z9.d}, p3, [x1, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: st3d {z7.d, z8.d, z9.d}, p3, [x1, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

st3d { z0.d, z1.d, z2.d }, p0, [x0, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: st3d { z0.d, z1.d, z2.d }, p0, [x0, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3d { z0.d, z1.d, z2.d }, p0, [x0, xzr]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: st3d { z0.d, z1.d, z2.d }, p0, [x0, xzr]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3d { z0.d, z1.d, z2.d }, p0, [x0, x0, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: st3d { z0.d, z1.d, z2.d }, p0, [x0, x0, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3d { z0.d, z1.d, z2.d }, p0, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: st3d { z0.d, z1.d, z2.d }, p0, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3d { z0.d, z1.d, z2.d }, p0, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: st3d { z0.d, z1.d, z2.d }, p0, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// error: restricted predicate has range [0, 7].

st3d {z2.d, z3.d, z4.d}, p8, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: st3d {z2.d, z3.d, z4.d}, p8, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

st3d { }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: st3d { }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3d { z0.d, z1.d, z2.d, z3.d }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st3d { z0.d, z1.d, z2.d, z3.d }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3d { z0.d, z1.d, z2.b }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: st3d { z0.d, z1.d, z2.b }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3d { z0.d, z1.d, z3.d }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must be sequential
// CHECK-NEXT: st3d { z0.d, z1.d, z3.d }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3d { v0.2d, v1.2d, v2.2d }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st3d { v0.2d, v1.2d, v2.2d }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z21.d, p5/z, z28.d
st3d    { z21.d, z22.d, z23.d }, p5, [x10, #15, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: st3d    { z21.d, z22.d, z23.d }, p5, [x10, #15, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z21, z28
st3d    { z21.d, z22.d, z23.d }, p5, [x10, #15, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: st3d    { z21.d, z22.d, z23.d }, p5, [x10, #15, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
