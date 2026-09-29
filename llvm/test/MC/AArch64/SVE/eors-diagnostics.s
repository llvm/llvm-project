// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Predicate register must have .b suffix

eors p0.h, p0/z, p0.h, p1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: eors p0.h, p0/z, p0.h, p1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

eors p0.s, p0/z, p0.s, p1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: eors p0.s, p0/z, p0.s, p1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

eors p0.d, p0/z, p0.d, p1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: eors p0.d, p0/z, p0.d, p1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Operation only has zeroing predicate behaviour (p0/z).

eors p0.b, p0/m, p1.b, p2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: eors p0.b, p0/m, p1.b, p2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
