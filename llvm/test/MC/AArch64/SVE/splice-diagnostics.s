// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Tied operands must match

splice  z0.b, p0, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: splice  z0.b, p0, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element widths.

splice  z0.b, p0, z0.b, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: splice  z0.b, p0, z0.b, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate

splice  z0.b, p8, z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: splice  z0.b, p8, z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

splice  z0.b, p7.b, z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: splice  z0.b, p7.b, z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

splice  z0.b, p7.q, z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: splice  z0.b, p7.q, z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z4.d, p7/z, z6.d
splice  z4.d, p7, z4.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: splice  z4.d, p7, z4.d, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
