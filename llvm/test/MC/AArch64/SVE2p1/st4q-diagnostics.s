// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid predicate register

st4q {z0.q, z1.q, z2.q, z3.q}, p8/z, [z0.d, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st4q {z0.q, z1.q, z2.q, z3.q}, p8/z, [z0.d, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4q {z23.q, z24.q, z25.q, z26.q}, p2/m, [x0, x0, lsl #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: st4q {z23.q, z24.q, z25.q, z26.q}, p2/m, [x0, x0, lsl #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4q {z21.q, z22.q, z23.q, z24.q}, p2.q, [x10, x21, lsl #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st4q {z21.q, z22.q, z23.q, z24.q}, p2.q, [x10, x21, lsl #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate offset

st4q {z23.q, z24.q, z25.q, z26.q}, p3, [x13, #-33, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28].
// CHECK-NEXT: st4q {z23.q, z24.q, z25.q, z26.q}, p3, [x13, #-33, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st4q {z23.q, z24.q, z25.q, z26.q}, p3, [x13, #29, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28].
// CHECK-NEXT: st4q {z23.q, z24.q, z25.q, z26.q}, p3, [x13, #29, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
