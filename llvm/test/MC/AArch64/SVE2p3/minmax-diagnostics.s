// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3,+fpv2 2>&1 < %s | FileCheck %s

fmaxinm z0.h, p0/m, z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: fmaxinm z0.h, p0/m, z1.h, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmininm z0.b, p0/m, z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmininm z0.b, p0/m, z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmaxmg z0.h, p0/m, z0.h, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmaxmg z0.h, p0/m, z0.h, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fminmg z0.h, p8/m, z0.h, z1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fminmg z0.h, p8/m, z0.h, z1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmaxmginm z0.h, p0/z, z0.h, z1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmaxmginm z0.h, p0/z, z0.h, z1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fminmginm z0.s, p0/m, z0.s, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fminmginm z0.s, p0/m, z0.s, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid reduction operands

fmaxinmv h0, p8, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fmaxinmv h0, p8, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmininmv b0, p0, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmininmv b0, p0, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmaxmgv z0.d, p0, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmaxmgv z0.d, p0, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fminmgv h0, p0, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fminmgv h0, p0, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmaxmginmv h0, p0/m, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmaxmginmv h0, p0/m, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fminmginmv v0.8h, p0, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fminmginmv v0.8h, p0, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
