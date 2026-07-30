// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid scalar registers

ctermne w30, wsp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ctermne w30, wsp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ctermne w30, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ctermne w30, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ctermne wsp, w30
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ctermne wsp, w30
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ctermne x0, w30
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ctermne x0, w30
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
