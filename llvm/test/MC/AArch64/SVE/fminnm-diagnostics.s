// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid immediates (must be 0.0 or 1.0)

fminnm z0.h, p0/m, z0.h, #0.5
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.0 or 1.0.
// CHECK-NEXT: fminnm z0.h, p0/m, z0.h, #0.5
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fminnm z0.h, p0/m, z0.h, #-0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.0 or 1.0.
// CHECK-NEXT: fminnm z0.h, p0/m, z0.h, #-0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fminnm z0.h, p0/m, z0.h, #0.0000000000000000000000001
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.0 or 1.0.
// CHECK-NEXT: fminnm z0.h, p0/m, z0.h, #0.0000000000000000000000001
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fminnm z0.h, p0/m, z0.h, #1.0000000000000000000000001
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.0 or 1.0.
// CHECK-NEXT: fminnm z0.h, p0/m, z0.h, #1.0000000000000000000000001
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fminnm z0.h, p0/m, z0.h, #0.9999999999999999999999999
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.0 or 1.0.
// CHECK-NEXT: fminnm z0.h, p0/m, z0.h, #0.9999999999999999999999999
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Tied operands must match

fminnm    z0.h, p7/m, z1.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: fminnm    z0.h, p7/m, z1.h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element widths.

fminnm    z0.b, p7/m, z0.b, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fminnm    z0.b, p7/m, z0.b, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fminnm    z0.h, p7/m, z0.h, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fminnm    z0.h, p7/m, z0.h, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate

fminnm    z0.h, p8/m, z0.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fminnm    z0.h, p8/m, z0.h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
