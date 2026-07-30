// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid scalar registers

ctermeq w30, wsp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ctermeq w30, wsp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ctermeq w30, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ctermeq w30, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ctermeq wsp, w30
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ctermeq wsp, w30
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ctermeq x0, w30
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ctermeq x0, w30
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
