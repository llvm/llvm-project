// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

frecpe    z0.b, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: frecpe    z0.b, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
frecpe   z0.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: frecpe   z0.d, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
frecpe   z0.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: frecpe   z0.d, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
