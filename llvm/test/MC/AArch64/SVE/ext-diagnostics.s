// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Tied operands must match

ext z0.b, z1.b, z2.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: ext z0.b, z1.b, z2.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element widths.

ext z0.h, z0.h, z1.h, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ext z0.h, z0.h, z1.h, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid immediate range.

ext z0.b, z0.b, z1.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255].
// CHECK-NEXT: ext z0.b, z0.b, z1.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ext z0.b, z0.b, z1.b, #256
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255].
// CHECK-NEXT: ext z0.b, z0.b, z1.b, #256
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.b, p0/z, z6.b
ext z31.b, z31.b, z0.b, #255
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: ext z31.b, z31.b, z0.b, #255
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
