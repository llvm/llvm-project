// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

frecpe    z0.b, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: frecpe    z0.b, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: