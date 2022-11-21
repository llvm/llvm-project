// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid predicate register

ld1q {z0.q}, p8/z, [z0.d, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: ld1q {z0.q}, p8/z, [z0.d, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1q {z23.q}, p2/m, [z3.d]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ld1q {z23.q}, p2/m, [z3.d]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1q {z21.q}, p2.q, [z5.d]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: ld1q {z21.q}, p2.q, [z5.d]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid order of base & offset

ld1q {z0.q}, p0/z, [x0, z0.d]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ld1q {z0.q}, p0/z, [x0, z0.d]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid general purpose register

ld1q {z0.q}, p0/z, [z0.d, sp]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ld1q {z0.q}, p0/z, [z0.d, sp]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid suffixes

ld1q {z0.q}, p0/z, [z2.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ld1q {z0.q}, p0/z, [z2.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
