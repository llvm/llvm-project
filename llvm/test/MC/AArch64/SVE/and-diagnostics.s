// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate not compatible with encode/decode function.

and z5.b, z5.b, #0xfa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: and z5.b, z5.b, #0xfa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

and z5.b, z5.b, #0xfff9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: and z5.b, z5.b, #0xfff9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

and z5.h, z5.h, #0xfffa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: and z5.h, z5.h, #0xfffa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

and z5.h, z5.h, #0xfffffff9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: and z5.h, z5.h, #0xfffffff9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

and z5.s, z5.s, #0xfffffffa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: and z5.s, z5.s, #0xfffffffa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

and z5.s, z5.s, #0xffffffffffffff9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: and z5.s, z5.s, #0xffffffffffffff9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

and z15.d, z15.d, #0xfffffffffffffffa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: and z15.d, z15.d, #0xfffffffffffffffa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Source and Destination Registers must match

and z7.d, z8.d, #254
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: and z7.d, z8.d, #254
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

and z0.d, p0/m, z1.d, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: and z0.d, p0/m, z1.d, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Element size specifiers should match.
and z21.d, z5.d, z26.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: and z21.d, z5.d, z26.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Predicate out of restricted predicate range

and z0.d, p8/z, z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: and z0.d, p8/z, z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Predicate register must have .b suffix

and p0.h, p0/z, p0.h, p1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: and p0.h, p0/z, p0.h, p1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

and p0.s, p0/z, p0.s, p1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: and p0.s, p0/z, p0.s, p1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

and p0.d, p0/z, p0.d, p1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: and p0.d, p0/z, p0.d, p1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Operation only has zeroing predicate behaviour (p0/z).

and p0.b, p0/m, p1.b, p2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: and p0.b, p0/m, p1.b, p2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
and     z0.d, z0.d, #0x6
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: and     z0.d, z0.d, #0x6
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z23.d, p0/z, z30.d
and     z23.d, z13.d, z8.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: and     z23.d, z13.d, z8.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z23, z30
and     z23.d, z13.d, z8.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: and     z23.d, z13.d, z8.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
