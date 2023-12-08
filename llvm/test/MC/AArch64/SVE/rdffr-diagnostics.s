// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid element widths

rdffr   p0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK: rdffr   p0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

rdffr   p0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK: rdffr   p0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

rdffr   p0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK: rdffr   p0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
