// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of lower bound [-128, 112].

ld1rqh z0.h, p0/z, [x0, #-144]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 16 in range [-128, 112].
// CHECK-NEXT: ld1rqh z0.h, p0/z, [x0, #-144]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqh z0.h, p0/z, [x0, #-129]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 16 in range [-128, 112].
// CHECK-NEXT: ld1rqh z0.h, p0/z, [x0, #-129]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqh z0.h, p0/z, [x0, #113]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 16 in range [-128, 112].
// CHECK-NEXT: ld1rqh z0.h, p0/z, [x0, #113]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqh z0.h, p0/z, [x0, #128]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 16 in range [-128, 112].
// CHECK-NEXT: ld1rqh z0.h, p0/z, [x0, #128]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqh z0.h, p0/z, [x0, #12]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 16 in range [-128, 112].
// CHECK-NEXT: ld1rqh z0.h, p0/z, [x0, #12]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediate suffix

ld1rqh z0.h, p0/z, [x0, #16, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1rqh z0.h, p0/z, [x0, #16, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid destination register width.

ld1rqh z0.b, p0/z, [x0, x1, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1rqh z0.b, p0/z, [x0, x1, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqh z0.s, p0/z, [x0, x1, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1rqh z0.s, p0/z, [x0, x1, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqh z0.d, p0/z, [x0, x1, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1rqh z0.d, p0/z, [x0, x1, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

ld1rqh z0.h, p0/z, [x0, xzr, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
// CHECK-NEXT: ld1rqh z0.h, p0/z, [x0, xzr, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqh z0.h, p0/z, [x0, x1, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
// CHECK-NEXT: ld1rqh z0.h, p0/z, [x0, x1, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqh z0.h, p0/z, [x0, w1, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
// CHECK-NEXT: ld1rqh z0.h, p0/z, [x0, w1, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqh z0.h, p0/z, [x0, w1, uxtw #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
// CHECK-NEXT: ld1rqh z0.h, p0/z, [x0, w1, uxtw #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
