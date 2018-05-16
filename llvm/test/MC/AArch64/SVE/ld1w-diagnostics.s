// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of lower bound [-8, 7].

ld1w z30.s, p6/z, [x25, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1w z30.s, p6/z, [x25, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z29.s, p5/z, [x15, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1w z29.s, p5/z, [x15, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z28.d, p2/z, [x28, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1w z28.d, p2/z, [x28, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z27.d, p1/z, [x26, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1w z27.d, p1/z, [x26, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

ld1w z12.s, p8/z, [x13, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld1w z12.s, p8/z, [x13, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z4.d, p8/z, [x11, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld1w z4.d, p8/z, [x11, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

ld1w { }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: ld1w { }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w { z1.s, z2.s }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1w { z1.s, z2.s }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w { v0.2d }, p0/z, [x1, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1w { v0.2d }, p0/z, [x1, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

ld1w z0.s, p0/z, [x0, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: ld1w z0.s, p0/z, [x0, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.s, p0/z, [x0, xzr]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: ld1w z0.s, p0/z, [x0, xzr]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.s, p0/z, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: ld1w z0.s, p0/z, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.s, p0/z, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: ld1w z0.s, p0/z, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.s, p0/z, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: ld1w z0.s, p0/z, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + vector addressing modes

ld1w z0.d, p0/z, [x0, z0.h]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1w z0.d, p0/z, [x0, z0.h]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.d, p0/z, [x0, z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1w z0.d, p0/z, [x0, z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.s, p0/z, [x0, z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw)'
// CHECK-NEXT: ld1w z0.s, p0/z, [x0, z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.s, p0/z, [x0, z0.s, uxtw #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw) #2'
// CHECK-NEXT: ld1w z0.s, p0/z, [x0, z0.s, uxtw #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.s, p0/z, [x0, z0.s, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw) #2'
// CHECK-NEXT: ld1w z0.s, p0/z, [x0, z0.s, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.d, p0/z, [x0, z0.d, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #2'
// CHECK-NEXT: ld1w z0.d, p0/z, [x0, z0.d, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.d, p0/z, [x0, z0.d, sxtw #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #2'
// CHECK-NEXT: ld1w z0.d, p0/z, [x0, z0.d, sxtw #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector + immediate addressing modes

ld1w z0.s, p0/z, [z0.s, #-4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: ld1w z0.s, p0/z, [z0.s, #-4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.s, p0/z, [z0.s, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: ld1w z0.s, p0/z, [z0.s, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.s, p0/z, [z0.s, #125]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: ld1w z0.s, p0/z, [z0.s, #125]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.s, p0/z, [z0.s, #128]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: ld1w z0.s, p0/z, [z0.s, #128]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.s, p0/z, [z0.s, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: ld1w z0.s, p0/z, [z0.s, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.d, p0/z, [z0.d, #-4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: ld1w z0.d, p0/z, [z0.d, #-4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.d, p0/z, [z0.d, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: ld1w z0.d, p0/z, [z0.d, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.d, p0/z, [z0.d, #125]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: ld1w z0.d, p0/z, [z0.d, #125]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.d, p0/z, [z0.d, #128]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: ld1w z0.d, p0/z, [z0.d, #128]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1w z0.d, p0/z, [z0.d, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: ld1w z0.d, p0/z, [z0.d, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
