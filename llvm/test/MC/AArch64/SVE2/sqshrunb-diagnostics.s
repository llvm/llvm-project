// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s

sqshrunb z30.b, z10.h, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8]
// CHECK-NEXT: sqshrunb z30.b, z10.h, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrunb z18.b, z27.h, #9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8]
// CHECK-NEXT: sqshrunb z18.b, z27.h, #9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrunb z26.h, z4.s, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqshrunb z26.h, z4.s, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrunb z25.h, z10.s, #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqshrunb z25.h, z10.s, #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrunb z17.s, z0.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32]
// CHECK-NEXT: sqshrunb z17.s, z0.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrunb z0.s, z15.d, #33
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32]
// CHECK-NEXT: sqshrunb z0.s, z15.d, #33
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid element width

sqshrunb z0.b, z0.b, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqshrunb z0.b, z0.b, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrunb z0.h, z0.h, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqshrunb z0.h, z0.h, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrunb z0.s, z0.s, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqshrunb z0.s, z0.s, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrunb z0.d, z0.d, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqshrunb z0.d, z0.d, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31, z6
sqshrunb     z31.s, z31.d, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqshrunb     z31.s, z31.d, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
