// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid immediate (multiple of 4 in range [0, 252]).

ld1rd z0.d, p1/z, [x0, #-8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 504].
// CHECK-NEXT: ld1rd z0.d, p1/z, [x0, #-8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rd z0.d, p1/z, [x0, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 504].
// CHECK-NEXT: ld1rd z0.d, p1/z, [x0, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rd z0.d, p1/z, [x0, #505]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 504].
// CHECK-NEXT: ld1rd z0.d, p1/z, [x0, #505]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rd z0.d, p1/z, [x0, #512]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 504].
// CHECK-NEXT: ld1rd z0.d, p1/z, [x0, #512]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rd z0.d, p1/z, [x0, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 8 in range [0, 504].
// CHECK-NEXT: ld1rd z0.d, p1/z, [x0, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid result vector element size

ld1rd z0.b, p1/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ld1rd z0.b, p1/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rd z0.h, p1/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ld1rd z0.h, p1/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rd z0.s, p1/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ld1rd z0.s, p1/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

ld1rd z0.d, p8/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld1rd z0.d, p8/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
