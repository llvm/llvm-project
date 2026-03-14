// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid predicate register

st2q {z0.q, z1.q}, p8, [z0.d, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st2q {z0.q, z1.q}, p8, [z0.d, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2q {z23.q, z24.q}, p2/m, [x0, x0, lsl #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: st2q {z23.q, z24.q}, p2/m, [x0, x0, lsl #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2q {z21.q, z22.q}, p2.q, [x10, x21, lsl #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st2q {z21.q, z22.q}, p2.q, [x10, x21, lsl #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate offset

st2q {z23.q, z24.q}, p3, [x13, #-17, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: st2q {z23.q, z24.q}, p3, [x13, #-17, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st2q {z23.q, z24.q}, p3, [x13, #15, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [-16, 14].
// CHECK-NEXT: st2q {z23.q, z24.q}, p3, [x13, #15, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
