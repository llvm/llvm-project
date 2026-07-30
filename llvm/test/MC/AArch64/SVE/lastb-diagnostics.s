// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid predicate

lastb   w0, p8, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: lastb   w0, p8, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lastb   w0, p7.b, w0, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: lastb   w0, p7.b, w0, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lastb   w0, p7.q, w0, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: lastb   w0, p7.q, w0, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element width

lastb   x0, p7, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lastb   x0, p7, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lastb   x0, p7, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lastb   x0, p7, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lastb   x0, p7, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lastb   x0, p7, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lastb   w0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lastb   w0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lastb   b0, p7, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lastb   b0, p7, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lastb   h0, p7, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lastb   h0, p7, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lastb   s0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lastb   s0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lastb   d0, p7, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lastb   d0, p7, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p7/z, z6.d
lastb   x0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: lastb   x0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
lastb   x0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: lastb   x0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31.d, p7/z, z6.d
lastb   d0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: lastb   d0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
lastb   d0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: lastb   d0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
