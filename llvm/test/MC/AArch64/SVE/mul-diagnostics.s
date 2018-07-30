// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid immediate range

mul z0.b, z0.b, #-129
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-128, 127].
// CHECK-NEXT: mul z0.b, z0.b, #-129
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mul z0.b, z0.b, #128
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-128, 127].
// CHECK-NEXT: mul z0.b, z0.b, #128
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Tied operands must match

mul z0.b, z1.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: mul z0.b, z1.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mul z0.b, p7/m, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: mul z0.b, p7/m, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate

mul z0.b, p8/m, z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: mul z0.b, p8/m, z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
mul z31.d, z31.d, #127
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: mul z31.d, z31.d, #127
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
