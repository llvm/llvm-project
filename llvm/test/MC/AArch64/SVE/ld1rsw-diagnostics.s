// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid immediate (multiple of 4 in range [0, 252]).

ld1rsw z0.d, p1/z, [x0, #-4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 252].
// CHECK-NEXT: ld1rsw z0.d, p1/z, [x0, #-4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rsw z0.d, p1/z, [x0, #256]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 252].
// CHECK-NEXT: ld1rsw z0.d, p1/z, [x0, #256]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rsw z0.d, p1/z, [x0, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 252].
// CHECK-NEXT: ld1rsw z0.d, p1/z, [x0, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid result vector element size

ld1rsw z0.b, p1/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1rsw z0.b, p1/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rsw z0.h, p1/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1rsw z0.h, p1/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rsw z0.s, p1/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld1rsw z0.s, p1/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

ld1rsw z0.d, p8/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld1rsw z0.d, p8/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
