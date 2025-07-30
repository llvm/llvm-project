// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid element width

frint32x     z31.b, p7/m, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: frint32x     z31.b, p7/m, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

frint32x     z31.h, p7/m, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: frint32x     z31.h, p7/m, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

frint32x     z31.s, p7/m, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: frint32x     z31.s, p7/m, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

frint32x     z31.d, p7/m, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: frint32x     z31.d, p7/m, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid predicate

frint32x     z31.s, p8/m, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: frint32x     z31.s, p8/m, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
