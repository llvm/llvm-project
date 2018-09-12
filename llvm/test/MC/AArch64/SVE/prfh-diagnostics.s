// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// invalid/missing predicate operation specifier

prfh p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: prefetch hint expected
// CHECK-NEXT: prfh p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #16, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: prefetch operand out of range, [0,15] expected
// CHECK-NEXT: prfh #16, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh plil1keep, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: prefetch hint expected
// CHECK-NEXT: prfh plil1keep, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #pldl1keep, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate value expected for prefetch operand
// CHECK-NEXT: prfh #pldl1keep, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// invalid scalar + scalar addressing modes

prfh #0, p0, [x0, #-33, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: prfh #0, p0, [x0, #-33, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [x0, #32, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: prfh #0, p0, [x0, #32, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
// CHECK-NEXT: prfh #0, p0, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [x0, x0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
// CHECK-NEXT: prfh #0, p0, [x0, x0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [x0, x0, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
// CHECK-NEXT: prfh #0, p0, [x0, x0, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + vector addressing modes

prfh #0, p0, [x0, z0.h]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: prfh #0, p0, [x0, z0.h]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [x0, z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw) #1'
// CHECK-NEXT: prfh #0, p0, [x0, z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [x0, z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw) #1'
// CHECK-NEXT: prfh #0, p0, [x0, z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [x0, z0.s, uxtw #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw) #1'
// CHECK-NEXT: prfh #0, p0, [x0, z0.s, uxtw #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [x0, z0.s, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw) #1'
// CHECK-NEXT: prfh #0, p0, [x0, z0.s, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [x0, z0.d, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #1'
// CHECK-NEXT: prfh #0, p0, [x0, z0.d, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [x0, z0.d, sxtw #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #1'
// CHECK-NEXT: prfh #0, p0, [x0, z0.d, sxtw #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector + immediate addressing modes

prfh #0, p0, [z0.s, #-2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 62].
// CHECK-NEXT: prfh #0, p0, [z0.s, #-2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [z0.s, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 62].
// CHECK-NEXT: prfh #0, p0, [z0.s, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [z0.s, #63]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 62].
// CHECK-NEXT: prfh #0, p0, [z0.s, #63]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [z0.s, #64]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 62].
// CHECK-NEXT: prfh #0, p0, [z0.s, #64]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [z0.s, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 62].
// CHECK-NEXT: prfh #0, p0, [z0.s, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [z0.d, #-2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 62].
// CHECK-NEXT: prfh #0, p0, [z0.d, #-2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [z0.d, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 62].
// CHECK-NEXT: prfh #0, p0, [z0.d, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [z0.d, #63]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 62].
// CHECK-NEXT: prfh #0, p0, [z0.d, #63]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [z0.d, #64]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 62].
// CHECK-NEXT: prfh #0, p0, [z0.d, #64]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfh #0, p0, [z0.d, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 62].
// CHECK-NEXT: prfh #0, p0, [z0.d, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// invalid predicate

prfh #0, p8, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: prfh #0, p8, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
prfh    pldl1keep, p0, [x0, z0.d, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: prfh    pldl1keep, p0, [x0, z0.d, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
prfh    pldl1keep, p0, [x0, z0.d, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: prfh    pldl1keep, p0, [x0, z0.d, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
