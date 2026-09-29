// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid predicate register

ld1d {z0.q}, p8/z, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: ld1d {z0.q}, p8/z, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d {z23.q}, p2/m, [x13, #-8, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ld1d {z23.q}, p2/m, [x13, #-8, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d {z23.q}, p2.q, [x13, #-8, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: ld1d {z23.q}, p2.q, [x13, #-8, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate range

ld1d {z0.q}, p0/z, [x0, #-9, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1d {z0.q}, p0/z, [x0, #-9, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d {z3.q}, p0/z, [x0, #8, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: ld1d {z3.q}, p0/z, [x0, #8, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
