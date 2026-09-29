// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate not compatible with encode/decode function.

orn z5.b, z5.b, #0xfa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: orn z5.b, z5.b, #0xfa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

orn z5.b, z5.b, #0xfff9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: orn z5.b, z5.b, #0xfff9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

orn z5.h, z5.h, #0xfffa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: orn z5.h, z5.h, #0xfffa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

orn z5.h, z5.h, #0xfffffff9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: orn z5.h, z5.h, #0xfffffff9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

orn z5.s, z5.s, #0xfffffffa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: orn z5.s, z5.s, #0xfffffffa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

orn z5.s, z5.s, #0xffffffffffffff9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: orn z5.s, z5.s, #0xffffffffffffff9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

orn z15.d, z15.d, #0xfffffffffffffffa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: orn z15.d, z15.d, #0xfffffffffffffffa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Source and Destination Registers must match

orn z7.d, z8.d, #254
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: orn z7.d, z8.d, #254
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

orn z7.d, z8.d, #254
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: orn z7.d, z8.d, #254
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Predicate register must have .b suffix

orn p0.h, p0/z, p0.h, p1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: orn p0.h, p0/z, p0.h, p1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

orn p0.s, p0/z, p0.s, p1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: orn p0.s, p0/z, p0.s, p1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

orn p0.d, p0/z, p0.d, p1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: orn p0.d, p0/z, p0.d, p1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Operation only has zeroing predicate behaviour (p0/z).

orn p0.b, p0/m, p1.b, p2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: orn p0.b, p0/m, p1.b, p2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
orn     z0.d, z0.d, #0x6
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: orn     z0.d, z0.d, #0x6
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
