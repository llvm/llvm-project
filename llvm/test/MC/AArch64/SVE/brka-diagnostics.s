// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Only .b is supported

brka  p0.s, p15/z, p15.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: brka  p0.s, p15/z, p15.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
