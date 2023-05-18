// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+f32mm,+f64mm  2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// FMMLA (SVE)

// Invalid element size

fmmla z0.h, z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width

// Mis-matched element size

fmmla z0.d, z1.s, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
fmmla z0.s, z1.d, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
fmmla z0.s, z1.s, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width


// --------------------------------------------------------------------------//
// LD1RO (SVE, scalar plus immediate)

// Immediate too high (>224)
ld1rob { z0.b }, p1/z, [x2, #256]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 32 in range [-256, 224].
ld1roh { z0.h }, p1/z, [x2, #256]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 32 in range [-256, 224].
ld1row { z0.s }, p1/z, [x2, #256]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 32 in range [-256, 224].
ld1rod { z0.d }, p1/z, [x2, #256]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 32 in range [-256, 224].

// Immediate too low (<-256)
ld1rob { z0.b }, p1/z, [x2, #-288]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 32 in range [-256, 224].
ld1roh { z0.h }, p1/z, [x2, #-288]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 32 in range [-256, 224].
ld1row { z0.s }, p1/z, [x2, #-288]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 32 in range [-256, 224].
ld1rod { z0.d }, p1/z, [x2, #-288]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 32 in range [-256, 224].

// Immediate not a multiple of 32
ld1rob { z0.b }, p1/z, [x2, #16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 32 in range [-256, 224].
ld1roh { z0.h }, p1/z, [x2, #16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 32 in range [-256, 224].
ld1row { z0.s }, p1/z, [x2, #16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 32 in range [-256, 224].
ld1rod { z0.d }, p1/z, [x2, #16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 32 in range [-256, 224].

// Prediate register too high
ld1rob { z0.b }, p8/z, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
ld1roh { z0.h }, p8/z, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
ld1row { z0.s }, p8/z, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
ld1rod { z0.d }, p8/z, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)


// --------------------------------------------------------------------------//
// LD1RO (SVE, scalar plus scalar)

// Shift amount not matched to data width
ld1rob { z0.b }, p1/z, [x2, x3, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
ld1roh { z0.h }, p1/z, [x2, x3, lsl #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #1'
ld1row { z0.s }, p1/z, [x2, x3, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #2'
ld1rod { z0.d }, p1/z, [x2, x3, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 with required shift 'lsl #3'

// Prediate register too high
ld1rob { z0.b }, p8/z, [x2, x3, lsl #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
ld1roh { z0.h }, p8/z, [x2, x3, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
ld1row { z0.s }, p8/z, [x2, x3, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
ld1rod { z0.d }, p8/z, [x2, x3, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
