// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

fcmeq   p0.b, p0/z, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register 
// CHECK-NEXT: fcmeq   p0.b, p0/z, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcmeq   p0.s, p0/z, z0.s, #1.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected floating-point constant #0.0
// CHECK-NEXT: fcmeq   p0.s, p0/z, z0.s, #1.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
