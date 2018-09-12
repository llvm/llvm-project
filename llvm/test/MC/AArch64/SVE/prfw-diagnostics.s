// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// invalid/missing predicate operation specifier

prfw p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: prefetch hint expected
// CHECK-NEXT: prfw p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #16, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: prefetch operand out of range, [0,15] expected
// CHECK-NEXT: prfw #16, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw plil1keep, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: prefetch hint expected
// CHECK-NEXT: prfw plil1keep, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #pldl1keep, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate value expected for prefetch operand
// CHECK-NEXT: prfw #pldl1keep, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// invalid scalar + scalar addressing modes

prfw #0, p0, [x0, #-33, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: prfw #0, p0, [x0, #-33, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [x0, #32, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: prfw #0, p0, [x0, #32, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: prfw #0, p0, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [x0, x0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: prfw #0, p0, [x0, x0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [x0, x0, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
// CHECK-NEXT: prfw #0, p0, [x0, x0, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + vector addressing modes

prfw #0, p0, [x0, z0.h]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: prfw #0, p0, [x0, z0.h]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [x0, z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw) #2'
// CHECK-NEXT: prfw #0, p0, [x0, z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [x0, z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw) #2'
// CHECK-NEXT: prfw #0, p0, [x0, z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [x0, z0.s, uxtw #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw) #2'
// CHECK-NEXT: prfw #0, p0, [x0, z0.s, uxtw #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [x0, z0.s, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw) #2'
// CHECK-NEXT: prfw #0, p0, [x0, z0.s, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [x0, z0.d, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #2'
// CHECK-NEXT: prfw #0, p0, [x0, z0.d, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [x0, z0.d, sxtw #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #2'
// CHECK-NEXT: prfw #0, p0, [x0, z0.d, sxtw #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector + immediate addressing modes

prfw #0, p0, [z0.s, #-4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: prfw #0, p0, [z0.s, #-4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [z0.s, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: prfw #0, p0, [z0.s, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [z0.s, #125]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: prfw #0, p0, [z0.s, #125]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [z0.s, #128]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: prfw #0, p0, [z0.s, #128]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [z0.s, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: prfw #0, p0, [z0.s, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [z0.d, #-4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: prfw #0, p0, [z0.d, #-4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [z0.d, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: prfw #0, p0, [z0.d, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [z0.d, #125]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: prfw #0, p0, [z0.d, #125]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [z0.d, #128]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: prfw #0, p0, [z0.d, #128]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfw #0, p0, [z0.d, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 124].
// CHECK-NEXT: prfw #0, p0, [z0.d, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// invalid predicate

prfw #0, p8, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: prfw #0, p8, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z8.d, p3/z, z15.d
prfw    #7, p3, [x13, z8.d, uxtw #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: prfw    #7, p3, [x13, z8.d, uxtw #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z8, z15
prfw    #7, p3, [x13, z8.d, uxtw #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: prfw    #7, p3, [x13, z8.d, uxtw #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z21.d, p5/z, z28.d
prfw    pldl3strm, p5, [x10, z21.d, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: prfw    pldl3strm, p5, [x10, z21.d, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z21, z28
prfw    pldl3strm, p5, [x10, z21.d, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: prfw    pldl3strm, p5, [x10, z21.d, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
