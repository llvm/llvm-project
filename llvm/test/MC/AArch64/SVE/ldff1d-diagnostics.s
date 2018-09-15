// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid operand (.b, .h, .s)

ldff1d z4.b, p7/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ldff1d z4.b, p7/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z4.h, p7/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ldff1d z4.h, p7/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z4.s, p7/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ldff1d z4.s, p7/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

ldff1d z4.d, p8/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ldff1d z4.d, p8/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

ldff1d z0.d, p0/z, [x0, sp]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 or xzr, with required shift 'lsl #3'
// CHECK-NEXT: ldff1d z0.d, p0/z, [x0, sp]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z0.d, p0/z, [x0, x0, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 or xzr, with required shift 'lsl #3'
// CHECK-NEXT: ldff1d z0.d, p0/z, [x0, x0, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z0.d, p0/z, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 or xzr, with required shift 'lsl #3'
// CHECK-NEXT: ldff1d z0.d, p0/z, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z0.d, p0/z, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 or xzr, with required shift 'lsl #3'
// CHECK-NEXT: ldff1d z0.d, p0/z, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + vector addressing modes

ldff1d z0.d, p0/z, [x0, z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ldff1d z0.d, p0/z, [x0, z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z0.d, p0/z, [x0, z0.d, uxtw #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #3'
// CHECK-NEXT: ldff1d z0.d, p0/z, [x0, z0.d, uxtw #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z0.d, p0/z, [x0, z0.d, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #3'
// CHECK-NEXT: ldff1d z0.d, p0/z, [x0, z0.d, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z0.d, p0/z, [x0, z0.d, lsl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected #imm after shift specifier
// CHECK-NEXT: ldff1d z0.d, p0/z, [x0, z0.d, lsl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector + immediate addressing modes

ldff1d z0.s, p0/z, [z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ldff1d z0.s, p0/z, [z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z0.s, p0/z, [z0.s, #8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ldff1d z0.s, p0/z, [z0.s, #8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z0.d, p0/z, [z0.d, #-8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 248].
// CHECK-NEXT: ldff1d z0.d, p0/z, [z0.d, #-8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z0.d, p0/z, [z0.d, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 248].
// CHECK-NEXT: ldff1d z0.d, p0/z, [z0.d, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z0.d, p0/z, [z0.d, #249]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 248].
// CHECK-NEXT: ldff1d z0.d, p0/z, [z0.d, #249]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z0.d, p0/z, [z0.d, #256]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 248].
// CHECK-NEXT: ldff1d z0.d, p0/z, [z0.d, #256]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1d z0.d, p0/z, [z0.d, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 248].
// CHECK-NEXT: ldff1d z0.d, p0/z, [z0.d, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
ldff1d  { z0.d }, p0/z, [z0.d]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ldff1d  { z0.d }, p0/z, [z0.d]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
ldff1d  { z0.d }, p0/z, [z0.d]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ldff1d  { z0.d }, p0/z, [z0.d]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
