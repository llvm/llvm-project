// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Restricted predicate out of range.

cmphs p0.b, p8/z, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: cmphs p0.b, p8/z, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid predicate operation

cmphs p0.b, p0/m, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cmphs p0.b, p0/m, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid source registers

cmphs p0.b, p0/z, z0.b, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmphs p0.b, p0/z, z0.b, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmphs p0.h, p0/z, z0.h, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmphs p0.h, p0/z, z0.h, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmphs p0.s, p0/z, z0.s, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmphs p0.s, p0/z, z0.s, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmphs p0.d, p0/z, z0.d, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmphs p0.d, p0/z, z0.d, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmphs p0.b, p0/z, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmphs p0.b, p0/z, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmphs p0.h, p0/z, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmphs p0.h, p0/z, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmphs p0.s, p0/z, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmphs p0.s, p0/z, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmphs p0.d, p0/z, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmphs p0.d, p0/z, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediate range

cmphs p0.s, p0/z, z0.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 127].
// CHECK-NEXT: cmphs p0.s, p0/z, z0.s, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmphs p0.s, p0/z, z0.s, #128
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 127].
// CHECK-NEXT: cmphs p0.s, p0/z, z0.s, #128
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
cmphs   p0.d, p0/z, z0.d, #127
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: cmphs   p0.d, p0/z, z0.d, #127
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
cmphs   p0.d, p0/z, z0.d, #127
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: cmphs   p0.d, p0/z, z0.d, #127
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.s, p0/z, z7.s
cmphs   p0.s, p0/z, z0.s, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: cmphs   p0.s, p0/z, z0.s, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
cmphs   p0.s, p0/z, z0.s, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: cmphs   p0.s, p0/z, z0.s, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
