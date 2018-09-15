// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Immediate out of lower bound [-24, 21].

st3w {z12.s, z13.s, z14.s}, p4, [x12, #-27, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: st3w {z12.s, z13.s, z14.s}, p4, [x12, #-27, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3w {z7.s, z8.s, z9.s}, p3, [x1, #24, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: st3w {z7.s, z8.s, z9.s}, p3, [x1, #24, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Immediate not a multiple of three.

st3w {z12.s, z13.s, z14.s}, p4, [x12, #-7, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: st3w {z12.s, z13.s, z14.s}, p4, [x12, #-7, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3w {z7.s, z8.s, z9.s}, p3, [x1, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: st3w {z7.s, z8.s, z9.s}, p3, [x1, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

st3w { z0.s, z1.s, z2.s }, p0, [x0, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: st3w { z0.s, z1.s, z2.s }, p0, [x0, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3w { z0.s, z1.s, z2.s }, p0, [x0, xzr]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: st3w { z0.s, z1.s, z2.s }, p0, [x0, xzr]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3w { z0.s, z1.s, z2.s }, p0, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: st3w { z0.s, z1.s, z2.s }, p0, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3w { z0.s, z1.s, z2.s }, p0, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: st3w { z0.s, z1.s, z2.s }, p0, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3w { z0.s, z1.s, z2.s }, p0, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: st3w { z0.s, z1.s, z2.s }, p0, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// error: restricted predicate has range [0, 7].

st3w {z2.s, z3.s, z4.s}, p8, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: st3w {z2.s, z3.s, z4.s}, p8, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

st3w { }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: st3w { }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3w { z0.s, z1.s, z2.s, z3.s }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st3w { z0.s, z1.s, z2.s, z3.s }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3w { z0.s, z1.s, z2.d }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: st3w { z0.s, z1.s, z2.d }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3w { z0.s, z1.s, z3.s }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must be sequential
// CHECK-NEXT: st3w { z0.s, z1.s, z3.s }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3w { v0.4s, v1.4s, v2.4s }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st3w { v0.4s, v1.4s, v2.4s }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z21.s, p5/z, z28.s
st3w    { z21.s, z22.s, z23.s }, p5, [x10, #15, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: st3w    { z21.s, z22.s, z23.s }, p5, [x10, #15, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z21, z28
st3w    { z21.s, z22.s, z23.s }, p5, [x10, #15, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: st3w    { z21.s, z22.s, z23.s }, p5, [x10, #15, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
