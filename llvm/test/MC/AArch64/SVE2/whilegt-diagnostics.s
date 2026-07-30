// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid scalar registers

whilegt  p15.b, xzr, sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: whilegt  p15.b, xzr, sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

whilegt  p15.b, xzr, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: whilegt  p15.b, xzr, w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

whilegt  p15.b, w0, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: whilegt  p15.b, w0, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate

whilegt  p15, w0, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: whilegt  p15, w0, w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
