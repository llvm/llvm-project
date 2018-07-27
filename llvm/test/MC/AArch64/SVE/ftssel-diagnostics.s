// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

ftssel    z0.b, z1.b, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ftssel    z0.b, z1.b, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: