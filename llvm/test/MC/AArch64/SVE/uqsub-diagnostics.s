// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Register z32 does not exist.
uqsub z22.h, z10.h, z32.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqsub z22.h, z10.h, z32.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Invalid element kind.
uqsub z20.h, z2.h, z31.x
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: uqsub z20.h, z2.h, z31.x
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Element size specifiers should match.
uqsub z27.h, z11.h, z27.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: uqsub z27.h, z11.h, z27.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediates

uqsub     z0.b, z0.b, #0, lsl #8      // #0, lsl #8 is not valid for .b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: uqsub     z0.b, z0.b, #0, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsub     z0.b, z0.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: uqsub     z0.b, z0.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsub     z0.b, z0.b, #1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: uqsub     z0.b, z0.b, #1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsub     z0.b, z0.b, #256
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: uqsub     z0.b, z0.b, #256
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsub     z0.h, z0.h, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: uqsub     z0.h, z0.h, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsub     z0.h, z0.h, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: uqsub     z0.h, z0.h, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsub     z0.h, z0.h, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: uqsub     z0.h, z0.h, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsub     z0.s, z0.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: uqsub     z0.s, z0.s, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsub     z0.s, z0.s, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: uqsub     z0.s, z0.s, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsub     z0.s, z0.s, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: uqsub     z0.s, z0.s, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsub     z0.d, z0.d, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: uqsub     z0.d, z0.d, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsub     z0.d, z0.d, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: uqsub     z0.d, z0.d, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqsub     z0.d, z0.d, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: uqsub     z0.d, z0.d, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
uqsub     z31.d, z31.d, #65280
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: uqsub     z31.d, z31.d, #65280
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.d, p0/z, z7.d
uqsub     z0.d, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: uqsub     z0.d, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
uqsub     z0.d, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: uqsub     z0.d, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
