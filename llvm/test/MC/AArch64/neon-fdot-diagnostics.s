// RUN: not llvm-mc -triple=aarch64 -mattr=f16f32dot 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid operand

fdot v0.2s, v0.4b, v0.4b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot v0.2s, v0.4b, v0.4b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot v0.2b, v0.4b, v0.4b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot v0.2b, v0.4b, v0.4b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot v0.2s, v0.4s, v0.4s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot v0.2s, v0.4s, v0.4s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot v0.2h, v0.4h, v0.4h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot v0.2h, v0.4h, v0.4h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// fdot indexed

fdot v0.2s, v0.4b, v0.4b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot v0.2s, v0.4b, v0.4b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot v0.2b, v0.4b, v0.4b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot v0.2b, v0.4b, v0.4b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot v0.2s, v0.4s, v0.4s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot v0.2s, v0.4s, v0.4s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot v0.2h, v0.4h, v0.4h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot v0.2h, v0.4h, v0.4h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate range

fdot v0.2s, v0.4h, v0.2h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fdot v0.2s, v0.4h, v0.2h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot v0.2s, v0.4h, v0.2h[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fdot v0.2s, v0.4h, v0.2h[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
