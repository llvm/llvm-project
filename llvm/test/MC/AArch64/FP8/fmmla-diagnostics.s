// RUN: not llvm-mc -triple=aarch64 -mattr=+f8f16mm,+f8f32mm  2>&1 < %s| FileCheck %s

fmmla v0.4h, v1.16b, v2.16b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmmla v0.4h, v1.16b, v2.16b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla v0.8s, v1.16b, v2.16b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: fmmla v0.8s, v1.16b, v2.16b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla v0.4s, v1.4s, v2.4s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmmla v0.4s, v1.4s, v2.4s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla v0.8h, v1.8h, v2.8h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmmla v0.8h, v1.8h, v2.8h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla v0.16b, v1.16b, v2.16b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmmla v0.16b, v1.16b, v2.16b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla v0.d, v1.16b, v2.16b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmmla v0.d, v1.16b, v2.16b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla v0.2d, v1.16b, v2.16b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmmla v0.2d, v1.16b, v2.16b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla v0.8h, v1.8b, v2.8b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmmla v0.8h, v1.8b, v2.8b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla v0.4s, v1.8b, v2.8b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmmla v0.4s, v1.8b, v2.8b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
