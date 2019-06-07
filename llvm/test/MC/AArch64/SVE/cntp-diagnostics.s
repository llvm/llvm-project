// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

cntp  sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cntp  sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate operand

cntp  x0, p15, p0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: cntp  x0, p15, p0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntp  x0, p15.b, p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: cntp  x0, p15.b, p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntp  x0, p15.q, p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: cntp  x0, p15.q, p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
