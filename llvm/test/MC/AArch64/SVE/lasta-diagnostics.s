// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid predicate

lasta   w0, p8, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: lasta   w0, p8, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element width

lasta   x0, p7, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lasta   x0, p7, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lasta   x0, p7, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lasta   x0, p7, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lasta   x0, p7, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lasta   x0, p7, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lasta   w0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lasta   w0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lasta   b0, p7, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lasta   b0, p7, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lasta   h0, p7, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lasta   h0, p7, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lasta   s0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lasta   s0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lasta   d0, p7, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: lasta   d0, p7, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p7/z, z6.d
lasta   x0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: lasta   x0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
lasta   x0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: lasta   x0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31.d, p7/z, z6.d
lasta   d0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: lasta   d0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
lasta   d0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: lasta   d0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
