// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of lower bound [-8, 7].

st1w z19.s, p2, [x18, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1w z19.s, p2, [x18, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Immediate out of upper bound [-8, 7].
st1w z1.s, p5, [x23, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1w z1.s, p5, [x23, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Immediate out of lower bound [-8, 7].
st1w z21.d, p2, [x29, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1w z21.d, p2, [x29, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Immediate out of upper bound [-8, 7].
st1w z10.d, p5, [x26, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1w z10.d, p5, [x26, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Restricted predicate has range [0, 7].

st1w z1.s, p8, [x3, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: st1w z1.s, p8, [x3, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z12.d, p8, [x26, #3, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: st1w z12.d, p8, [x26, #3, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

st1w { }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: st1w { }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w { z1.s, z2.s }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1w { z1.s, z2.s }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w { v0.4s }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1w { v0.4s }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

st1w z0.s, p0, [x0, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: st1w z0.s, p0, [x0, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.s, p0, [x0, xzr]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: st1w z0.s, p0, [x0, xzr]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.s, p0, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: st1w z0.s, p0, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.s, p0, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: st1w z0.s, p0, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.s, p0, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: st1w z0.s, p0, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + vector addressing modes

st1w z0.d, p0, [x0, z0.h]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1w z0.d, p0, [x0, z0.h]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.d, p0, [x0, z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1w z0.d, p0, [x0, z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.s, p0, [x0, z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw)'
// CHECK-NEXT: st1w z0.s, p0, [x0, z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.s, p0, [x0, z0.s, uxtw #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw) #2'
// CHECK-NEXT: st1w z0.s, p0, [x0, z0.s, uxtw #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.s, p0, [x0, z0.s, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw) #2'
// CHECK-NEXT: st1w z0.s, p0, [x0, z0.s, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.d, p0, [x0, z0.d, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #2'
// CHECK-NEXT: st1w z0.d, p0, [x0, z0.d, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.d, p0, [x0, z0.d, sxtw #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #2'
// CHECK-NEXT: st1w z0.d, p0, [x0, z0.d, sxtw #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector + immediate addressing modes

st1w z0.s, p0, [z0.s, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: st1w z0.s, p0, [z0.s, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.s, p0, [z0.s, #-4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: st1w z0.s, p0, [z0.s, #-4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.s, p0, [z0.s, #125]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: st1w z0.s, p0, [z0.s, #125]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.s, p0, [z0.s, #128]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: st1w z0.s, p0, [z0.s, #128]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.s, p0, [z0.s, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: st1w z0.s, p0, [z0.s, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.d, p0, [z0.d, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: st1w z0.d, p0, [z0.d, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.d, p0, [z0.d, #-4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: st1w z0.d, p0, [z0.d, #-4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.d, p0, [z0.d, #125]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: st1w z0.d, p0, [z0.d, #125]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.d, p0, [z0.d, #128]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: st1w z0.d, p0, [z0.d, #128]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1w z0.d, p0, [z0.d, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: st1w z0.d, p0, [z0.d, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
