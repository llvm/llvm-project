// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Element sizes must match

sadalp z0.b, p0/m, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sadalp z0.b, p0/m, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sadalp z0.h, p0/m, z1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sadalp z0.h, p0/m, z1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sadalp z0.s, p0/m, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sadalp z0.s, p0/m, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sadalp z0.d, p0/m, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sadalp z0.d, p0/m, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Predicate not in restricted predicate range

sadalp z0.h, p8/m, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: sadalp z0.h, p8/m, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative movprfx tests

movprfx z31.s, p0/z, z6.s // element type of the source operand, rather than destination.
sadalp z31.d, p0/m, z30.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx with a different element size
// CHECK-NEXT: sadalp z31.d, p0/m, z30.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
