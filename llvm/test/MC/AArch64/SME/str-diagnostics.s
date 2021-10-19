// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid matrix operand (expected: za)

str za0h.b[w12, #0], [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za
// CHECK-NEXT: str za0h.b[w12, #0], [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

str za3.s[w12, #0], [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za
// CHECK-NEXT: str za3.s[w12, #0], [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid vector select register (expected: w12-w15)

str za[w11, #0], [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w12, w15]
// CHECK-NEXT: str za[w11, #0], [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

str za[w16, #0], [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w12, w15]
// CHECK-NEXT: str za[w16, #0], [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid vector select offset (expected: 0-15)

str za[w12, #16], [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15].
// CHECK-NEXT: str za[w12, #16], [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid memory operands

str za[w12, #0], [w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: str za[w12, #0], [w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

str za[w12, #0], [x0, #16, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15].
// CHECK-NEXT: str za[w12, #0], [x0, #16, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

str za[w12, #0], [x0, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: str za[w12, #0], [x0, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
