// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Out of range index offset

movaz {z0.h-z1.h}, za0h.h[w12, 1:2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 2 in the range [0, 6] or [0, 14] depending on the instruction, and the second immediate is immf + 1.
// CHECK-NEXT: movaz {z0.h-z1.h}, za0h.h[w12, 1:2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz {z0.b-z3.b}, za0v.b[w12, 1:4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 4 in the range [0, 4] or [0, 12] depending on the instruction, and the second immediate is immf + 3.
// CHECK-NEXT: movaz {z0.b-z3.b}, za0v.b[w12, 1:4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz {z0.s-z1.s}, za0h.s[w12, 0:2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: movaz {z0.s-z1.s}, za0h.s[w12, 0:2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz {z0.d-z3.d}, za0h.d[w12, 0:4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: movaz {z0.d-z3.d}, za0h.d[w12, 0:4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz  {z4.d-z7.d}, za.d[w9, 8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: movaz  {z4.d-z7.d}, za.d[w9, 8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz  {z4.d-z7.d}, za.d[w9, -1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: movaz  {z4.d-z7.d}, za.d[w9, -1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz z1.q, za1h.q[w12, 1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be 0.
// CHECK-NEXT: movaz z1.q, za1h.q[w12, 1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz z31.h, za1h.h[w15, 8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: movaz z31.h, za1h.h[w15, 8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz z31.h, za1h.h[w15, -1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: movaz z31.h, za1h.h[w15, -1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz z2.b, za0v.b[w15, -1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15].
// CHECK-NEXT: movaz z2.b, za0v.b[w15, -1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz z2.b, za0v.b[w15, 16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15].
// CHECK-NEXT: movaz z2.b, za0v.b[w15, 16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz z31.s, za1h.s[w15, 4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 3].
// CHECK-NEXT: movaz z31.s, za1h.s[w15, 4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz z31.s, za1h.s[w15, -1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 3].
// CHECK-NEXT: movaz z31.s, za1h.s[w15, -1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz z31.d, za1v.d[w15, 2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 1].
// CHECK-NEXT: movaz z31.d, za1v.d[w15, 2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz z31.d, za1h.d[w15, -1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 1].
// CHECK-NEXT: movaz z31.d, za1h.d[w15, -1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

movaz z0.h, za0v.h[w11, 0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w12, w15]
// CHECK-NEXT: movaz z0.h, za0v.h[w11, 0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movaz z0.h, za0v.h[w16, 0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w12, w15]
// CHECK-NEXT: movaz z0.h, za0v.h[w16, 0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid matrix operand

movaz z31.s, za1h.d[w15, -1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-3]h.s or za[0-3]v.s
// CHECK-NEXT: movaz z31.s, za1h.d[w15, -1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
