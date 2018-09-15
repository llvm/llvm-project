// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid predicate

clastb   w0, p8, w0, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: clastb   w0, p8, w0, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element width

clastb   w0, p7, x0, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: clastb   w0, p7, x0, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clastb   x0, p7, x0, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clastb   x0, p7, x0, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clastb   x0, p7, x0, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clastb   x0, p7, x0, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clastb   x0, p7, x0, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clastb   x0, p7, x0, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clastb   w0, p7, w0, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clastb   w0, p7, w0, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clastb   b0, p7, b0, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clastb   b0, p7, b0, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clastb   h0, p7, h0, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clastb   h0, p7, h0, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clastb   s0, p7, s0, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clastb   s0, p7, s0, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clastb   d0, p7, d0, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clastb   d0, p7, d0, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clastb   z0.b, p7, z0.b, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clastb   z0.b, p7, z0.b, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clastb   z0.h, p7, z0.h, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clastb   z0.h, p7, z0.h, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clastb   z0.s, p7, z0.s, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clastb   z0.s, p7, z0.s, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clastb   z0.d, p7, z0.d, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clastb   z0.d, p7, z0.d, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p7/z, z6.d
clastb   x0, p7, x0, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: clastb   x0, p7, x0, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
clastb   x0, p7, x0, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: clastb   x0, p7, x0, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31.d, p7/z, z6.d
clastb   d0, p7, d0, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: clastb   d0, p7, d0, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
clastb   d0, p7, d0, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: clastb   d0, p7, d0, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.d, p7/z, z7.d
clastb   z0.d, p7, z0.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: clastb   z0.d, p7, z0.d, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
