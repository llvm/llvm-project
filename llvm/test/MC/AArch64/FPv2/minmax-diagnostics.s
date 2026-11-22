// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+fpv2 2>&1 < %s | FileCheck %s

fmaxinm h0, s1, h2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmaxinm h0, s1, h2
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmininm s0, s1, d2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmininm s0, s1, d2
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmaxmg d0, h1, d2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmaxmg d0, h1, d2
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fminmg b0, b1, b2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fminmg b0, b1, b2
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmaxmginm b0, b1, b2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmaxmginm b0, b1, b2
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fminmginm b0, b1, b2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fminmginm b0, b1, b2
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
