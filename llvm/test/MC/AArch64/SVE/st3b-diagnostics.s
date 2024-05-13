// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Immediate out of lower bound [-24, 21].

st3b {z12.b, z13.b, z14.b}, p4, [x12, #-27, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: st3b {z12.b, z13.b, z14.b}, p4, [x12, #-27, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3b {z7.b, z8.b, z9.b}, p3, [x1, #24, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: st3b {z7.b, z8.b, z9.b}, p3, [x1, #24, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Immediate not a multiple of three.

st3b {z12.b, z13.b, z14.b}, p4, [x12, #-7, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: st3b {z12.b, z13.b, z14.b}, p4, [x12, #-7, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3b {z7.b, z8.b, z9.b}, p3, [x1, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: st3b {z7.b, z8.b, z9.b}, p3, [x1, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

st3b { z0.b, z1.b, z2.b }, p0, [x0, xzr]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: st3b { z0.b, z1.b, z2.b }, p0, [x0, xzr]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3b { z0.b, z1.b, z2.b }, p0, [x0, x0, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: st3b { z0.b, z1.b, z2.b }, p0, [x0, x0, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3b { z0.b, z1.b, z2.b }, p0, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: st3b { z0.b, z1.b, z2.b }, p0, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3b { z0.b, z1.b, z2.b }, p0, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: st3b { z0.b, z1.b, z2.b }, p0, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid predicate

st3b {z2.b, z3.b, z4.b}, p8, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st3b {z2.b, z3.b, z4.b}, p8, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3b {z2.b, z3.b, z4.b}, p7.b, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st3b {z2.b, z3.b, z4.b}, p7.b, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3b {z2.b, z3.b, z4.b}, p7.q, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st3b {z2.b, z3.b, z4.b}, p7.q, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

st3b { }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: st3b { }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3b { z0.b, z1.b, z2.b, z3.b }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st3b { z0.b, z1.b, z2.b, z3.b }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3b { z0.b, z1.b, z2.h }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: st3b { z0.b, z1.b, z2.h }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3b { z0.b, z1.b, z3.b }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must have the same sequential stride
// CHECK-NEXT: st3b { z0.b, z1.b, z3.b }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st3b { v0.16b, v1.16b, v2.16b }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st3b { v0.16b, v1.16b, v2.16b }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z21.b, p5/z, z28.b
st3b    { z21.b, z22.b, z23.b }, p5, [x10, #15, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: st3b    { z21.b, z22.b, z23.b }, p5, [x10, #15, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z21, z28
st3b    { z21.b, z22.b, z23.b }, p5, [x10, #15, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: st3b    { z21.b, z22.b, z23.b }, p5, [x10, #15, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
