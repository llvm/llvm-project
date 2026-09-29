// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Restricted predicate out of range.

cmpeq p0.b, p8/z, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: cmpeq p0.b, p8/z, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid predicate operation

cmpeq p0.b, p0/m, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cmpeq p0.b, p0/m, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid source registers

cmpeq p0.b, p0/z, z0.b, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpeq p0.b, p0/z, z0.b, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpeq p0.h, p0/z, z0.h, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpeq p0.h, p0/z, z0.h, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpeq p0.s, p0/z, z0.s, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpeq p0.s, p0/z, z0.s, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpeq p0.d, p0/z, z0.d, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpeq p0.d, p0/z, z0.d, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpeq p0.b, p0/z, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpeq p0.b, p0/z, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpeq p0.h, p0/z, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpeq p0.h, p0/z, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpeq p0.s, p0/z, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpeq p0.s, p0/z, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpeq p0.d, p0/z, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpeq p0.d, p0/z, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediate range

cmpeq p0.s, p0/z, z0.s, #-17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-16, 15].
// CHECK-NEXT: cmpeq p0.s, p0/z, z0.s, #-17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpeq p0.s, p0/z, z0.s, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-16, 15].
// CHECK-NEXT: cmpeq p0.s, p0/z, z0.s, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
cmpeq   p0.d, p0/z, z0.d, #15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: cmpeq   p0.d, p0/z, z0.d, #15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
cmpeq   p0.d, p0/z, z0.d, #15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: cmpeq   p0.d, p0/z, z0.d, #15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.s, p0/z, z7.s
cmpeq   p0.s, p0/z, z0.s, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: cmpeq   p0.s, p0/z, z0.s, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
cmpeq   p0.s, p0/z, z0.s, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: cmpeq   p0.s, p0/z, z0.s, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
