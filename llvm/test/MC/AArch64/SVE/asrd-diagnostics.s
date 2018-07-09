// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

asrd z18.b, p0/m, z28.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8]
// CHECK-NEXT: asrd z18.b, p0/m, z28.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

asrd z1.b, p0/m, z9.b, #9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8]
// CHECK-NEXT: asrd z1.b, p0/m, z9.b, #9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

asrd z21.h, p0/m, z2.h, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: asrd z21.h, p0/m, z2.h, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

asrd z14.h, p0/m, z30.h, #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: asrd z14.h, p0/m, z30.h, #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

asrd z6.s, p0/m, z12.s, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32]
// CHECK-NEXT: asrd z6.s, p0/m, z12.s, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

asrd z23.s, p0/m, z19.s, #33
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32]
// CHECK-NEXT: asrd z23.s, p0/m, z19.s, #33
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

asrd z3.d, p0/m, z24.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]
// CHECK-NEXT: asrd z3.d, p0/m, z24.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

asrd z25.d, p0/m, z16.d, #65
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]
// CHECK-NEXT: asrd z25.d, p0/m, z16.d, #65
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
