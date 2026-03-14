// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of lower bound [-128, 112].

ld1rqd z0.d, p0/z, [x0, #-144]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 16 in range [-128, 112].
// CHECK-NEXT: ld1rqd z0.d, p0/z, [x0, #-144]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqd z0.d, p0/z, [x0, #-129]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 16 in range [-128, 112].
// CHECK-NEXT: ld1rqd z0.d, p0/z, [x0, #-129]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqd z0.d, p0/z, [x0, #113]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 16 in range [-128, 112].
// CHECK-NEXT: ld1rqd z0.d, p0/z, [x0, #113]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqd z0.d, p0/z, [x0, #128]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 16 in range [-128, 112].
// CHECK-NEXT: ld1rqd z0.d, p0/z, [x0, #128]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqd z0.d, p0/z, [x0, #12]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 16 in range [-128, 112].
// CHECK-NEXT: ld1rqd z0.d, p0/z, [x0, #12]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediate suffix

ld1rqd z0.d, p0/z, [x0, #16, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1rqd z0.d, p0/z, [x0, #16, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid destination register width.

ld1rqd z0.b, p0/z, [x0, x1, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ld1rqd z0.b, p0/z, [x0, x1, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqd z0.h, p0/z, [x0, x1, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ld1rqd z0.h, p0/z, [x0, x1, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqd z0.s, p0/z, [x0, x1, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ld1rqd z0.s, p0/z, [x0, x1, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

ld1rqd z0.d, p0/z, [x0, xzr, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: ld1rqd z0.d, p0/z, [x0, xzr, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqd z0.d, p0/z, [x0, x1, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: ld1rqd z0.d, p0/z, [x0, x1, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqd z0.d, p0/z, [x0, w1, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: ld1rqd z0.d, p0/z, [x0, w1, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rqd z0.d, p0/z, [x0, w1, uxtw #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'
// CHECK-NEXT: ld1rqd z0.d, p0/z, [x0, w1, uxtw #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z23.d, p3/z, z30.d
ld1rqd  { z23.d }, p3/z, [x13, #112]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ld1rqd  { z23.d }, p3/z, [x13, #112]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z23, z30
ld1rqd  { z23.d }, p3/z, [x13, #112]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ld1rqd  { z23.d }, p3/z, [x13, #112]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
