
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-lutv2  2>&1 < %s | FileCheck %s
// --------------------------------------------------------------------------//
// Invalid vector select register
movt   z0, z31
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid lookup table, expected zt0
// CHECK-NEXT: movt   z0, z31
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
// --------------------------------------------------------------------------//
// Invalid vector select offset
//
movt    zt0[-1, mul vl], z31
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 3].
// CHECK-NEXT: movt    zt0[-1, mul vl], z31
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
movt    zt0[4, mul vl], z31
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 3].
// CHECK-NEXT: movt    zt0[4, mul vl], z31
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
// --------------------------------------------------------------------------//
// Invalid mul vl
movt  zt0[0, mul vl 3],  z0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: ']' expected
// CHECK-NEXT: movt  zt0[0, mul vl 3],  z0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
movt  zt0[0, mul #4],  z0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: movt  zt0[0, mul #4],  z0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
