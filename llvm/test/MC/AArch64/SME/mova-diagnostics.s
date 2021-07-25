// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid predicate (expected: p0-p7)

mova z0.b, p8/m, za0h.b[w12, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: mova z0.b, p8/m, za0h.b[w12, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid tile

// tile-to-vector

mova z0.b, p0/m, za1h.b[w12, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
// CHECK-NEXT: mova z0.b, p0/m, za1h.b[w12, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.h, p0/m, za2h.h[w12, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
// CHECK-NEXT: mova z0.h, p0/m, za2h.h[w12, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.s, p0/m, za4h.s[w12, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
// CHECK-NEXT: mova z0.s, p0/m, za4h.s[w12, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.d, p0/m, za8h.d[w12, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
// CHECK-NEXT: mova z0.d, p0/m, za8h.d[w12, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.q, p0/m, za16h.q[w12, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
// CHECK-NEXT: mova z0.q, p0/m, za16h.q[w12, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// vector-to-tile

mova za1h.b[w12, #0], p0/m, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
// CHECK-NEXT: mova za1h.b[w12, #0], p0/m, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za2h.h[w12, #0], p0/m, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
// CHECK-NEXT: mova za2h.h[w12, #0], p0/m, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za4h.s[w12, #0], p0/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
// CHECK-NEXT: mova za4h.s[w12, #0], p0/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za8h.d[w12, #0], p0/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
// CHECK-NEXT: mova za8h.d[w12, #0], p0/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za16h.q[w12, #0], p0/m, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
// CHECK-NEXT: mova za16h.q[w12, #0], p0/m, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid matrix operand

// tile-to-vector

mova z0.b, p0/m, za0h.h[w12, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za0h.b or za0v.b
// CHECK-NEXT: mova z0.b, p0/m, za0h.h[w12, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.h, p0/m, za[w12, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-1]h.h or za[0-1]v.h
// CHECK-NEXT: mova z0.h, p0/m, za[w12, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.s, p0/m, za2.s[w12, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-3]h.s or za[0-3]v.s
// CHECK-NEXT: mova z0.s, p0/m, za2.s[w12, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.d, p0/m, za2v.s[w12, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-7]h.d or za[0-7]v.d
// CHECK-NEXT: mova z0.d, p0/m, za2v.s[w12, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.q, p0/m, za0h.b[w12, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-15]h.q or za[0-15]v.q
// CHECK-NEXT: mova z0.q, p0/m, za0h.b[w12, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// vector-to-tile, only one test here since the intended instruction variant is
// ambiguous when failing to match on the first operand.

mova za[w12, #0], p0/m, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-7]h.d or za[0-7]v.d
// CHECK-NEXT: mova za[w12, #0], p0/m, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid vector select register (expected: w12-w15)

mova z0.h, p0/m, za0h.h[w11, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w12, w15]
// CHECK-NEXT: mova z0.h, p0/m, za0h.h[w11, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.s, p0/m, za0h.s[w16, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w12, w15]
// CHECK-NEXT: mova z0.s, p0/m, za0h.s[w16, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za0h.d[w11, #0], p0/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w12, w15]
// CHECK-NEXT: mova za0h.d[w11, #0], p0/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za0h.q[w16, #0], p0/m, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w12, w15]
// CHECK-NEXT: mova za0h.q[w16, #0], p0/m, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid vector select offset
//
//   expected: .b => 0-15, .h => 0-7, .s => 0-3, .d => 0-1, .q => NONE

// tile-to-vector

mova z0.b, p0/m, za0h.b[w12, #16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15].
// CHECK-NEXT: mova z0.b, p0/m, za0h.b[w12, #16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.h, p0/m, za0h.h[w12, #8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: mova z0.h, p0/m, za0h.h[w12, #8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.s, p0/m, za0h.s[w12, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 3].
// CHECK-NEXT: mova z0.s, p0/m, za0h.s[w12, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.d, p0/m, za0h.d[w12, #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 1].
// CHECK-NEXT: mova z0.d, p0/m, za0h.d[w12, #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova z0.q, p0/m, za0h.q[w12, #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mova z0.q, p0/m, za0h.q[w12, #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// vector-to-tile

mova za0h.b[w12, #16], p0/m, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15].
// CHECK-NEXT: mova za0h.b[w12, #16], p0/m, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za0h.h[w12, #8], p0/m, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: mova za0h.h[w12, #8], p0/m, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za0h.s[w12, #4], p0/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 3].
// CHECK-NEXT: mova za0h.s[w12, #4], p0/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za0h.d[w12, #2], p0/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 1].
// CHECK-NEXT: mova za0h.d[w12, #2], p0/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za0h.q[w12, #0], p0/m, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mova za0h.q[w12, #0], p0/m, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid ZPR element width

mova za0h.b[w12, #0], p0/m, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: mova za0h.b[w12, #0], p0/m, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za0h.h[w12, #0], p0/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: mova za0h.h[w12, #0], p0/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za0h.s[w12, #0], p0/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: mova za0h.s[w12, #0], p0/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za0h.d[w12, #0], p0/m, z0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: mova za0h.d[w12, #0], p0/m, z0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mova za0h.q[w12], p0/m, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: mova za0h.q[w12], p0/m, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
