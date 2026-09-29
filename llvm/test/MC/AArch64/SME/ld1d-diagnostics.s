// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid tile (expected: za[0-7]h.d or za[0-7]v.d)

ld1d {za8h.d[w12, 0]}, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
// CHECK-NEXT: ld1d {za8h.d[w12, 0]}, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d {za[w12, 0]}, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-7]h.d or za[0-7]v.d
// CHECK-NEXT: ld1d {za[w12, 0]}, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d {za3h.s[w12, 0]}, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-7]h.d or za[0-7]v.d
// CHECK-NEXT: ld1d {za3h.s[w12, 0]}, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid vector select register (expected: w12-w15)

ld1d {za0h.d[w11, 0]}, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w12, w15]
// CHECK-NEXT: ld1d {za0h.d[w11, 0]}, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d {za0h.d[w16, 0]}, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w12, w15]
// CHECK-NEXT: ld1d {za0h.d[w16, 0]}, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid vector select offset (expected: 0-1)

ld1d {za0h.d[w12]}, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 1].
// CHECK-NEXT: ld1d {za0h.d[w12]}, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d {za0h.d[w12, 2]}, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 1].
// CHECK-NEXT: ld1d {za0h.d[w12, 2]}, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid predicate (expected: p0-p7)

ld1d {za0h.d[w12, 0]}, p8/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: ld1d {za0h.d[w12, 0]}, p8/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid predicate qualifier (expected: /z)

ld1d {za0h.d[w12, 0]}, p0/m, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ld1d {za0h.d[w12, 0]}, p0/m, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid memory operands

ld1d {za0h.d[w12, 0]}, p0/z, [w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ld1d {za0h.d[w12, 0]}, p0/z, [w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d {za0h.d[w12, 0]}, p0/z, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 or xzr, with required shift 'lsl #3'
// CHECK-NEXT: ld1d {za0h.d[w12, 0]}, p0/z, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1d {za0h.d[w12, 0]}, p0/z, [x0, x0, lsl #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 or xzr, with required shift 'lsl #3'
// CHECK-NEXT: ld1d {za0h.d[w12, 0]}, p0/z, [x0, x0, lsl #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
