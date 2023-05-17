// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid predicate register

umaxqv v0.2d, p11, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: umaxqv v0.2d, p11, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector register

umaxqv v0.4h, p1, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: umaxqv v0.4h, p1, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umaxqv z1.s, p1, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: umaxqv z1.s, p1, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

umaxqv v0.8h, p1, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: umaxqv v0.8h, p1, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
