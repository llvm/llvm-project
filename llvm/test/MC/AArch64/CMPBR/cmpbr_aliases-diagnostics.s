// RUN: not llvm-mc -triple=aarch64 -filetype=obj -mattr=+cmpbr 2>&1 < %s | FileCheck %s

L:
    cbge w5, #0, L
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]
    cbge x5, #65, L
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]

    cbhs w5, #0, L
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]
    cbhs x5, #65, L
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]

    cble w5, #-2, L
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-1, 62]
    cble x5, #63, L
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-1, 62]

    cbls w5, #-2, L
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-1, 62]
    cbls x5, #63, L
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-1, 62]

   cbls x3, w5, L
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-1, 62]