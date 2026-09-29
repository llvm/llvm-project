// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid immediates (must be 0.5 or 1.0)

fsub z0.h, p0/m, z0.h, #0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 1.0.
// CHECK-NEXT: fsub z0.h, p0/m, z0.h, #0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsub z0.h, p0/m, z0.h, #0.4999999999999999999999999
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 1.0.
// CHECK-NEXT: fsub z0.h, p0/m, z0.h, #0.4999999999999999999999999
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsub z0.h, p0/m, z0.h, #0.5000000000000000000000001
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 1.0.
// CHECK-NEXT: fsub z0.h, p0/m, z0.h, #0.5000000000000000000000001
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsub z0.h, p0/m, z0.h, #1.0000000000000000000000001
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 1.0.
// CHECK-NEXT: fsub z0.h, p0/m, z0.h, #1.0000000000000000000000001
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsub z0.h, p0/m, z0.h, #0.9999999999999999999999999
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 1.0.
// CHECK-NEXT: fsub z0.h, p0/m, z0.h, #0.9999999999999999999999999
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Tied operands must match

fsub    z0.h, p7/m, z1.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: fsub    z0.h, p7/m, z1.h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element widths.

fsub    z0.b, p7/m, z0.b, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fsub    z0.b, p7/m, z0.b, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsub    z0.h, p7/m, z0.h, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fsub    z0.h, p7/m, z0.h, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsub z0.b, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fsub z0.b, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fsub z0.h, z1.s, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fsub z0.h, z1.s, z2.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid predicate

fsub    z0.h, p8/m, z0.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fsub    z0.h, p8/m, z0.h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
fsub z0.d, z1.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fsub z0.d, z1.d, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
fsub z0.d, z1.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fsub z0.d, z1.d, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
