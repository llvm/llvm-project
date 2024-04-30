// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lut  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid lane indices

luti2 v2.16b, {v1.16b}, v0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti2 v2.16b, {v1.16b}, v0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 v3.16b, {v2.16b}, v1[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: luti2 v3.16b, {v2.16b}, v1[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 v30.8h, {v21.8h}, v20[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: luti2 v30.8h, {v21.8h}, v20[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 v31.8h, {v31.8h}, v31[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: luti2 v31.8h, {v31.8h}, v31[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

luti2 v2.8h, {v1.16b}, v0[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti2 v2.8h, {v1.16b}, v0[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti2 v31.16b, {v31.8h}, v31[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti2 v31.16b, {v31.8h}, v31[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
