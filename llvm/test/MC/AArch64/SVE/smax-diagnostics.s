// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

smax    z0.b, z0.b, #-129
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-128, 127].
// CHECK-NEXT: smax    z0.b, z0.b, #-129
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smax    z31.b, z31.b, #128
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-128, 127].
// CHECK-NEXT: smax    z31.b, z31.b, #128
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smax    z0.b, p8/m, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: smax    z0.b, p8/m, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
smax    z31.d, z31.d, #127
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: smax    z31.d, z31.d, #127
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
