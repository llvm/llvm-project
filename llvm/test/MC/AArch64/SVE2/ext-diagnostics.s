// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element widths.

ext z0.h, { z1.h, z2.h }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ext z0.h, { z1.h, z2.h }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ext z0.s, { z1.s, z2.s }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ext z0.s, { z1.s, z2.s }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ext z0.d, { z1.d, z2.d }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ext z0.d, { z1.d, z2.d }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid immediate range.

ext z0.b, { z1.b, z2.b }, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255].
// CHECK-NEXT: ext z0.b, { z1.b, z2.b }, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ext z0.b, { z1.b, z2.b }, #256
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255].
// CHECK-NEXT: ext z0.b, { z1.b, z2.b }, #256
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

ext z0.b, { }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: ext z0.b, { }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ext z0.b, { z1.b }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ext z0.b, { z1.b }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ext z0.b, { z1.b, z2.b, z3.b }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ext z0.b, { z1.b, z2.b, z3.b }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ext z0.b, { z1.b, z2.h }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: ext z0.b, { z1.b, z2.h }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ext z0.b, { z1.b, z31.b }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ext z0.b, { z1.b, z31.b }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ext z0.b, { v0.4b, v1.4b }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ext z0.b, { v0.4b, v1.4b }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31, z6
ext z31.b, { z30.b, z31.b }, #255
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ext z31.b, { z30.b, z31.b }, #255
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31.b, p0/z, z6.b
ext z31.b, { z30.b, z31.b }, #255
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ext z31.b, { z30.b, z31.b }, #255
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
