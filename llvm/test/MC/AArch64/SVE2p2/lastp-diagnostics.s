// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid predicate operand

lastp  x0, p15, p0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: lastp  x0, p15, p0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lastp  x0, p15.b, p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: lastp  x0, p15.b, p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lastp  x0, p15.q, p0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: lastp  x0, p15.q, p0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid register types

lastp  sp, p15, p0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: lastp  sp, p15, p0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lastp  w0, p15, p0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: lastp  w0, p15, p0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: