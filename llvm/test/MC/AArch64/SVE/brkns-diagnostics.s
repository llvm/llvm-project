// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// BRKN only supports merging predication

brkns  p0.b, p15/m, p1.b, p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: brkns  p0.b, p15/m, p1.b, p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Check tied operand constraints

brkns  p0.b, p15/z, p1.b, p1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: brkns  p0.b, p15/z, p1.b, p1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Only .b is supported

brkns  p15.s, p15/z, p15.s, p15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: brkns  p15.s, p15/z, p15.s, p15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
