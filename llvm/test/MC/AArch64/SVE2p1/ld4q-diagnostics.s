// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid predicate register

ld4q {z0.q, z1.q, z2.q, z3.q}, p8/z, [z0.d, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: ld4q {z0.q, z1.q, z2.q, z3.q}, p8/z, [z0.d, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld4q {z23.q, z24.q, z25.q, z26.q}, p2/m, [x0, x0, lsl #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ld4q {z23.q, z24.q, z25.q, z26.q}, p2/m, [x0, x0, lsl #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld4q {z21.q, z22.q, z23.q, z24.q}, p2.q, [x10, x21, lsl #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: ld4q {z21.q, z22.q, z23.q, z24.q}, p2.q, [x10, x21, lsl #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate offset

ld4q {z23.q, z24.q, z25.q, z26.q}, p3/z, [x13, #-33, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28].
// CHECK-NEXT: ld4q {z23.q, z24.q, z25.q, z26.q}, p3/z, [x13, #-33, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld4q {z23.q, z24.q, z25.q, z26.q}, p3/z, [x13, #29, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28].
// CHECK-NEXT: ld4q {z23.q, z24.q, z25.q, z26.q}, p3/z, [x13, #29, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
