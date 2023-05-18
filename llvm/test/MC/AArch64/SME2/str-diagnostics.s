// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2  2>&1 < %s| FileCheck %s

str zt, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid lookup table, expected zt0
// CHECK-NEXT: str zt, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
