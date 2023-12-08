// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
//  Invalid pattern name
// --------------------------------------------------------------------------//

ptrues p0.s, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ptrues p0.s, vl512
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

ptrues p0.s, vl9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ptrues p0.s, vl9
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
//  Invalid immediate range
// --------------------------------------------------------------------------//

ptrues p0.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: ptrues p0.s, #-1
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

ptrues p0.s, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: ptrues p0.s, #32
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:
