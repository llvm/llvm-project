// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+fp8 2>&1 < %s | FileCheck %s
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+fp8 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//

fcvtn z0.b, {z1.h, z2.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error:  Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT:  fcvtn z0.b, {z1.h, z2.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtn z0.h, {z0.h, z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtn z0.h, {z0.h, z1.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtn z0.b, {z0.b, z1.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtn z0.b, {z0.b, z1.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:



fcvtnb  z0.b, {z1.s, z2.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element type
// CHECK-NEXT:  fcvtnb z0.b, {z1.s, z2.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtnb z0.h, {z0.s, z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fcvtnb z0.h, {z0.s, z1.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtnb z0.b, {z0.h, z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtnb z0.b, {z0.h, z1.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:



bfcvtn z0.b, {z1.h, z2.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error:  Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT:  bfcvtn z0.b, {z1.h, z2.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfcvtn z0.h, {z0.h, z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfcvtn z0.h, {z0.h, z1.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfcvtn z0.b, {z0.b, z1.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfcvtn z0.b, {z0.b, z1.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:



fcvtnt  z0.b, {z1.s, z2.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element type
// CHECK-NEXT:  fcvtnt z0.b, {z1.s, z2.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtnt z0.h, {z0.s, z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtnt z0.h, {z0.s, z1.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtnt z0.b, {z0.h, z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtnt z0.b, {z0.h, z1.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: