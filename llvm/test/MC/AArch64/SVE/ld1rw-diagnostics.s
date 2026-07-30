// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid immediate (multiple of 4 in range [0, 252]).

ld1rw z0.s, p1/z, [x0, #-4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 252].
// CHECK-NEXT: ld1rw z0.s, p1/z, [x0, #-4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rw z0.s, p1/z, [x0, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 252].
// CHECK-NEXT: ld1rw z0.s, p1/z, [x0, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rw z0.s, p1/z, [x0, #253]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 252].
// CHECK-NEXT: ld1rw z0.s, p1/z, [x0, #253]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rw z0.s, p1/z, [x0, #256]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 252].
// CHECK-NEXT: ld1rw z0.s, p1/z, [x0, #256]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rw z0.s, p1/z, [x0, #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [0, 252].
// CHECK-NEXT: ld1rw z0.s, p1/z, [x0, #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid result vector element size

ld1rw z0.b, p1/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ld1rw z0.b, p1/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld1rw z0.h, p1/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ld1rw z0.h, p1/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

ld1rw z0.s, p8/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: ld1rw z0.s, p8/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p7/z, z6.d
ld1rw   { z31.d }, p7/z, [sp, #252]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ld1rw   { z31.d }, p7/z, [sp, #252]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
ld1rw   { z31.d }, p7/z, [sp, #252]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ld1rw   { z31.d }, p7/z, [sp, #252]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
