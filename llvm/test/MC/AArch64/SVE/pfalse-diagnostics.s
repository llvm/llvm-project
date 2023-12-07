// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Only .b is supported

pfalse p15.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: pfalse p15.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Support until pn15.b

pfalse pn16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: pfalse pn16.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
