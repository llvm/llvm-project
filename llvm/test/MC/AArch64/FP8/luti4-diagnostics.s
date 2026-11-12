// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lut  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid lane indices

luti4 v2.16b, {v1.16b}, v0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: luti4 v2.16b, {v1.16b}, v0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 v3.16b, {v2.16b}, v1[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: luti4 v3.16b, {v2.16b}, v1[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 v3.8h, {v0.8h, v1.8h}, v2[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti4 v3.8h, {v0.8h, v1.8h}, v2[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 v3.8h, {v0.8h, v1.8h}, v2[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti4 v3.8h, {v0.8h, v1.8h}, v2[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector lists

luti4 v30.8h, {v0.8h, v2.8h}, v3[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti4 v30.8h, {v0.8h, v2.8h}, v3[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

luti4 v2.8h, {v1.16b}, v0[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti4 v2.8h, {v1.16b}, v0[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 v31.8h, {v20.16b, v21.16b}, v31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti4 v31.8h,  {v20.16b, v21.16b}, v31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti4 v3.s, {v0.8h, v1.8h}, v2[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti4 v3.s, {v0.8h, v1.8h}, v2[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: