// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

lsl z18.b, z28.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7]
// CHECK-NEXT: lsl z18.b, z28.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z1.b, z9.b, #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7]
// CHECK-NEXT: lsl z1.b, z9.b, #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z21.h, z2.h, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15]
// CHECK-NEXT: lsl z21.h, z2.h, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z14.h, z30.h, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15]
// CHECK-NEXT: lsl z14.h, z30.h, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z6.s, z12.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31]
// CHECK-NEXT: lsl z6.s, z12.s, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z23.s, z19.s, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31]
// CHECK-NEXT: lsl z23.s, z19.s, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z3.d, z24.d, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63]
// CHECK-NEXT: lsl z3.d, z24.d, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

lsl z25.d, z16.d, #64
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63]
// CHECK-NEXT: lsl z25.d, z16.d, #64
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
