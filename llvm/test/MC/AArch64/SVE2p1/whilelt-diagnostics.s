// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid Pattern

whilelt pn8.b, x0, x0, vlx1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: whilelt pn8.b, x0, x0, vlx1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

whilelt pn8.b, x0, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// CHECK-NEXT: whilelt pn8.b, x0, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid use of predicate without suffix

whilelt pn8, x0, x0, vlx2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid predicate register, expected PN in range pn8..pn15 with element suffix.
// CHECK-NEXT: whilelt pn8, x0, x0, vlx2
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Out of range Predicate register

whilelt pn7.b, x0, x0, vlx2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid predicate register, expected PN in range pn8..pn15 with element suffix.
// CHECK-NEXT: whilelt pn7.b, x0, x0, vlx2
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

whilelt { p0.b, p2.b }, x13, x8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must be sequential
// CHECK-NEXT: whilelt { p0.b, p2.b }, x13, x8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

whilelt { p15.b, p0.b }, x13, x8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive predicate registers, where the first vector is a multiple of 2 and with correct element type
// CHECK-NEXT: whilelt { p15.b, p0.b }, x13, x8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
