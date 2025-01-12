// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid predicate operand

firstp  x0, p15, p0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: firstp  x0, p15, p0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

firstp  x0, p15.b, p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: firstp  x0, p15.b, p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

firstp  x0, p15.q, p0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: firstp  x0, p15.q, p0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid register types

firstp  sp, p15, p0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: firstp  sp, p15, p0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

firstp  w0, p15, p0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: firstp  w0, p15, p0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
