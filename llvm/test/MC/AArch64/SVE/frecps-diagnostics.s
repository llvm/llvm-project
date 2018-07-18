// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element size

frecps z0.b, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: frecps z0.b, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

frecps z0.h, z1.s, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: frecps z0.h, z1.s, z2.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
