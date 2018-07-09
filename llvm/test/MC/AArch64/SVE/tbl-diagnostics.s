// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

tbl z0.h, z0.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: tbl z0.h, z0.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

tbl { z0.h }, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: tbl { z0.h }, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
