// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+faminmax 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Element size extension incorrect

famax  v0.16s, v0.4s, v0.4s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: famax  v0.16s, v0.4s, v0.4s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famax  v0.4h, v0.4s, v0.4s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famax  v0.4h, v0.4s, v0.4s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famax  v0.8h, v0.8s, v0.8s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: famax  v0.8h, v0.8s, v0.8s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famax  v0.2s, v0.2h, v0.2h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famax  v0.2s, v0.2h, v0.2h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famax  v0.4s, v31.4h, v0.4h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famax  v0.4s, v31.4h, v0.4h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famax  v0.2d, v31.2h, v0.2h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famax  v0.2d, v31.2h, v0.2h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin  v0.16s, v0.4s, v0.4s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: famin  v0.16s, v0.4s, v0.4s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin  v0.4h, v0.4s, v0.4s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famin  v0.4h, v0.4s, v0.4s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin  v0.8h, v0.8s, v0.8s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: famin  v0.8h, v0.8s, v0.8s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin  v0.2s, v0.2h, v0.2h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famin  v0.2s, v0.2h, v0.2h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin  v0.4s, v31.4h, v0.4h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famin  v0.4s, v31.4h, v0.4h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin  v0.2d, v31.2h, v0.2h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famin  v0.2d, v31.2h, v0.2h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
