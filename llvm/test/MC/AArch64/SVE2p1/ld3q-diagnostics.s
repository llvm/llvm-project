// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid predicate register

ld3q {z0.q, z1.q, z2.q}, p8/z, [z0.d, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: ld3q {z0.q, z1.q, z2.q}, p8/z, [z0.d, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld3q {z23.q, z24.q, z25.q}, p2/m, [x0, x0, lsl #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ld3q {z23.q, z24.q, z25.q}, p2/m, [x0, x0, lsl #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld3q {z21.q, z22.q, z23.q}, p2.q, [x10, x21, lsl #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: ld3q {z21.q, z22.q, z23.q}, p2.q, [x10, x21, lsl #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate offset

ld3q {z23.q, z24.q, z25.q}, p3/z, [x13, #-25, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: ld3q {z23.q, z24.q, z25.q}, p3/z, [x13, #-25, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld3q {z23.q, z24.q, z25.q}, p3/z, [x13, #22, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: ld3q {z23.q, z24.q, z25.q}, p3/z, [x13, #22, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
