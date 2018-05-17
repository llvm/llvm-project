// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Immediate out of lower bound [-32, 28].

st4h {z12.h, z13.h, z14.h, z15.h}, p4, [x12, #-36, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28].
// CHECK-NEXT: st4h {z12.h, z13.h, z14.h, z15.h}, p4, [x12, #-36, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4h {z7.h, z8.h, z9.h, z10.h}, p3, [x1, #32, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28].
// CHECK-NEXT: st4h {z7.h, z8.h, z9.h, z10.h}, p3, [x1, #32, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Immediate not a multiple of four.

st4h {z12.h, z13.h, z14.h, z15.h}, p4, [x12, #-7, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28].
// CHECK-NEXT: st4h {z12.h, z13.h, z14.h, z15.h}, p4, [x12, #-7, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4h {z7.h, z8.h, z9.h, z10.h}, p3, [x1, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28].
// CHECK-NEXT: st4h {z7.h, z8.h, z9.h, z10.h}, p3, [x1, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
// CHECK-NEXT: st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, xzr]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
// CHECK-NEXT: st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, xzr]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, x0, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
// CHECK-NEXT: st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, x0, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
// CHECK-NEXT: st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
// CHECK-NEXT: st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// error: restricted predicate has range [0, 7].

st4h {z2.h, z3.h, z4.h, z5.h}, p8, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: st4h {z2.h, z3.h, z4.h, z5.h}, p8, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

st4h { }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: st4h { }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4h { z0.h, z1.h, z2.h, z3.h, z4.h }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: st4h { z0.h, z1.h, z2.h, z3.h, z4.h }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4h { z0.h, z1.h, z2.h, z3.s }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: st4h { z0.h, z1.h, z2.h, z3.s }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4h { z0.h, z1.h, z3.h, z5.h }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must be sequential
// CHECK-NEXT: st4h { z0.h, z1.h, z3.h, z5.h }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4h { v0.8h, v1.8h, v2.8h }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st4h { v0.8h, v1.8h, v2.8h }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
