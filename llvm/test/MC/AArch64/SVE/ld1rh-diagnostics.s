// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid immediate (multiple of 2 in range [0, 126]).

ld1rh z0.h, p1/z, [x0, #-2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 126].
// CHECK-NEXT: ld1rh z0.h, p1/z, [x0, #-2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rh z0.h, p1/z, [x0, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 126].
// CHECK-NEXT: ld1rh z0.h, p1/z, [x0, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rh z0.h, p1/z, [x0, #127]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 126].
// CHECK-NEXT: ld1rh z0.h, p1/z, [x0, #127]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rh z0.h, p1/z, [x0, #128]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 126].
// CHECK-NEXT: ld1rh z0.h, p1/z, [x0, #128]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rh z0.h, p1/z, [x0, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 2 in range [0, 126].
// CHECK-NEXT: ld1rh z0.h, p1/z, [x0, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid result vector element size

ld1rh z0.b, p1/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ld1rh z0.b, p1/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

ld1rh z0.h, p8/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld1rh z0.h, p8/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
