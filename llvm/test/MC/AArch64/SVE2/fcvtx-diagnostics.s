// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Invalid element width

fcvtx    z0.b, p0/m, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcvtx    z0.b, p0/m, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtx    z0.h, p0/m, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcvtx    z0.h, p0/m, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtx    z0.s, p0/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcvtx    z0.s, p0/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtx    z0.d, p0/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcvtx    z0.d, p0/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid predicate operation

fcvtx   z0.s, p0/z, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtx   z0.s, p0/z, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// error: invalid restricted predicate register, expected p0..p7 (without element suffix)

fcvtx    z0.s, p8/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fcvtx    z0.s, p8/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
