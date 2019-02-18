// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

lsl z18.b, z28.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7]
// CHECK-NEXT: lsl z18.b, z28.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z1.b, z9.b, #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7]
// CHECK-NEXT: lsl z1.b, z9.b, #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z18.b, p0/m, z28.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7]
// CHECK-NEXT: lsl z18.b, p0/m, z28.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z1.b, p0/m, z9.b, #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7]
// CHECK-NEXT: lsl z1.b, p0/m, z9.b, #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z21.h, z2.h, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15]
// CHECK-NEXT: lsl z21.h, z2.h, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z14.h, z30.h, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15]
// CHECK-NEXT: lsl z14.h, z30.h, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z21.h, p0/m, z2.h, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15]
// CHECK-NEXT: lsl z21.h, p0/m, z2.h, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z14.h, p0/m, z30.h, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15]
// CHECK-NEXT: lsl z14.h, p0/m, z30.h, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z6.s, z12.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31]
// CHECK-NEXT: lsl z6.s, z12.s, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z23.s, z19.s, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31]
// CHECK-NEXT: lsl z23.s, z19.s, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z6.s, p0/m, z12.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31]
// CHECK-NEXT: lsl z6.s, p0/m, z12.s, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z23.s, p0/m, z19.s, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31]
// CHECK-NEXT: lsl z23.s, p0/m, z19.s, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z3.d, z24.d, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63]
// CHECK-NEXT: lsl z3.d, z24.d, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z25.d, z16.d, #64
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63]
// CHECK-NEXT: lsl z25.d, z16.d, #64
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z3.d, p0/m, z24.d, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63]
// CHECK-NEXT: lsl z3.d, p0/m, z24.d, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z25.d, p0/m, z16.d, #64
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63]
// CHECK-NEXT: lsl z25.d, p0/m, z16.d, #64
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Source and Destination Registers must match

lsl z0.b, p0/m, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: lsl z0.b, p0/m, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z0.b, p0/m, z1.b, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: lsl z0.b, p0/m, z1.b, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Element sizes must match

lsl z0.b, z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lsl z0.b, z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z0.b, p0/m, z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lsl z0.b, p0/m, z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z0.b, p0/m, z0.b, z1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lsl z0.b, p0/m, z0.b, z1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Predicate not in restricted predicate range

lsl z0.b, p8/m, z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: lsl z0.b, p8/m, z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
lsl     z31.d, z31.d, #63
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: lsl     z31.d, z31.d, #63
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
lsl     z31.d, z31.d, #63
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: lsl     z31.d, z31.d, #63
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.s, p0/z, z7.s
lsl     z0.s, z1.s, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: lsl     z0.s, z1.s, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
lsl     z0.s, z1.s, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: lsl     z0.s, z1.s, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
