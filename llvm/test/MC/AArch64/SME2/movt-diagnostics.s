// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s| FileCheck %s

// index must be a multiple of 8 in range [0, 56].
// --------------------------------------------------------------------------//

movt x0, zt0[57]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 56].
// CHECK-NEXT: movt x0, zt0[57]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movt x0, zt0[58]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 56].
// CHECK-NEXT: movt x0, zt0[58]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movt x0, zt0[64]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 56].
// CHECK-NEXT: movt x0, zt0[64]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movt x0, zt0[72]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 56].
// CHECK-NEXT: movt x0, zt0[72]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid zt0 register

movt x0, zt1[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unexpected token in argument list
// CHECK-NEXT: movt x0, zt1[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
