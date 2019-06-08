// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Immediate out of lower bound [-16, 14].

st2d {z12.d, z13.d}, p4, [x12, #-18, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: st2d {z12.d, z13.d}, p4, [x12, #-18, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2d {z7.d, z8.d}, p3, [x1, #16, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: st2d {z7.d, z8.d}, p3, [x1, #16, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Immediate not a multiple of two.

st2d {z12.d, z13.d}, p4, [x12, #-7, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: st2d {z12.d, z13.d}, p4, [x12, #-7, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2d {z7.d, z8.d}, p3, [x1, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: st2d {z7.d, z8.d}, p3, [x1, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

st2d { z0.d, z1.d }, p0, [x0, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: st2d { z0.d, z1.d }, p0, [x0, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2d { z0.d, z1.d }, p0, [x0, xzr]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: st2d { z0.d, z1.d }, p0, [x0, xzr]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2d { z0.d, z1.d }, p0, [x0, x0, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: st2d { z0.d, z1.d }, p0, [x0, x0, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2d { z0.d, z1.d }, p0, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: st2d { z0.d, z1.d }, p0, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2d { z0.d, z1.d }, p0, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: st2d { z0.d, z1.d }, p0, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid predicate

st2d {z2.d, z3.d}, p8, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st2d {z2.d, z3.d}, p8, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2d {z2.d, z3.d}, p7.b, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st2d {z2.d, z3.d}, p7.b, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2d {z2.d, z3.d}, p7.q, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st2d {z2.d, z3.d}, p7.q, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

st2d { }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: st2d { }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2d { z0.d, z1.d, z2.d }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st2d { z0.d, z1.d, z2.d }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2d { z0.d, z1.b }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: st2d { z0.d, z1.b }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2d { z0.d, z2.d }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must be sequential
// CHECK-NEXT: st2d { z0.d, z2.d }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2d { v0.2d, v1.2d }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st2d { v0.2d, v1.2d }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z21.d, p5/z, z28.d
st2d    { z21.d, z22.d }, p5, [x10, #10, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: st2d    { z21.d, z22.d }, p5, [x10, #10, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z21, z28
st2d    { z21.d, z22.d }, p5, [x10, #10, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: st2d    { z21.d, z22.d }, p5, [x10, #10, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
