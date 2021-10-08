// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid predicate

// missing element type suffix
psel p0, p0, p0[w12]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register.
// CHECK-NEXT: psel p0, p0, p0[w12]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid index base register register (w12-w15)

psel p0, p0, p0.b[w11]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w12, w15]
// CHECK-NEXT: psel p0, p0, p0.b[w11]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

psel p0, p0, p0.b[w16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w12, w15]
// CHECK-NEXT: psel p0, p0, p0.b[w16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediates

psel p0, p0, p0.b[w12, #16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15].
// CHECK-NEXT: psel p0, p0, p0.b[w12, #16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

psel p0, p0, p0.h[w12, #8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: psel p0, p0, p0.h[w12, #8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

psel p0, p0, p0.s[w12, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 3].
// CHECK-NEXT: psel p0, p0, p0.s[w12, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

psel p0, p0, p0.d[w12, #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 1].
// CHECK-NEXT: psel  p0, p0, p0.d[w12, #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
