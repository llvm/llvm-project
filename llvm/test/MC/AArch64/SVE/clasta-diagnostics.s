// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid predicate

clasta   w0, p8, w0, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: clasta   w0, p8, w0, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element width

clasta   w0, p7, x0, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: clasta   w0, p7, x0, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clasta   x0, p7, x0, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clasta   x0, p7, x0, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clasta   x0, p7, x0, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clasta   x0, p7, x0, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clasta   x0, p7, x0, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clasta   x0, p7, x0, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clasta   w0, p7, w0, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clasta   w0, p7, w0, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clasta   b0, p7, b0, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clasta   b0, p7, b0, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clasta   h0, p7, h0, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clasta   h0, p7, h0, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clasta   s0, p7, s0, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clasta   s0, p7, s0, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clasta   d0, p7, d0, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clasta   d0, p7, d0, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clasta   z0.b, p7, z0.b, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clasta   z0.b, p7, z0.b, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clasta   z0.h, p7, z0.h, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clasta   z0.h, p7, z0.h, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clasta   z0.s, p7, z0.s, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clasta   z0.s, p7, z0.s, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

clasta   z0.d, p7, z0.d, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: clasta   z0.d, p7, z0.d, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
