// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Tied operands must match

pnext p0.b, p15, p1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: pnext p0.b, p15, p1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
