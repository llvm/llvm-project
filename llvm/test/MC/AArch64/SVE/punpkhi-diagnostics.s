// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element widths.

punpkhi p0.b, p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: punpkhi p0.b, p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

punpkhi p0.s, p0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: punpkhi p0.s, p0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
