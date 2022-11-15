// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Out of range index offset

zero za.d[w11, 8, vgx2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: zero za.d[w11, 8, vgx2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero za.d[w11, -1, vgx4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: zero za.d[w11, -1, vgx4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero za.d[w11, 5:8, vgx4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 4 in the range [0, 4] or [0, 12] depending on the instruction, and the second immediate is immf + 3.
// CHECK-NEXT: zero za.d[w11, 5:8, vgx4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero za.d[w11, 5:8, vgx2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 4 in the range [0, 4] or [0, 12] depending on the instruction, and the second immediate is immf + 3.
// CHECK-NEXT: zero za.d[w11, 5:8, vgx2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero za.d[w11, 0:4, vgx4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: zero za.d[w11, 0:4, vgx4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero za.d[w11, 0:4, vgx2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: zero za.d[w11, 0:4, vgx2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero za.d[w11, 11:15]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: zero za.d[w11, 11:15]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid select register

zero za.d[w7, 7, vgx2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: zero za.d[w7, 7, vgx2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

zero za.d[w12, 7, vgx2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: zero za.d[w12, 7, vgx2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid suffix

zero za.s[w11, 7, vgx2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .d
// CHECK-NEXT: zero za.s[w11, 7, vgx2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
