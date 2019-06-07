// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid predicate

ptest p15, p15.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: ptest p15, p15.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ptest p15.b, p15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: ptest p15.b, p15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ptest p15.q, p15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: ptest p15.q, p15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
