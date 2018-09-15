// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid scalar registers

whilelo  p15.b, xzr, sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: whilelo  p15.b, xzr, sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

whilelo  p15.b, xzr, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: whilelo  p15.b, xzr, w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

whilelo  p15.b, w0, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: whilelo  p15.b, w0, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
