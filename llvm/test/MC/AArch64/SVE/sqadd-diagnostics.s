// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Register z32 does not exist.
sqadd z22.h, z10.h, z32.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqadd z22.h, z10.h, z32.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Invalid element kind.
sqadd z20.h, z2.h, z31.x
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: sqadd z20.h, z2.h, z31.x
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Element size specifiers should match.
sqadd z27.h, z11.h, z27.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqadd z27.h, z11.h, z27.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediates

sqadd     z0.b, z0.b, #0, lsl #8      // #0, lsl #8 is not valid for .b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: sqadd     z0.b, z0.b, #0, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqadd     z0.b, z0.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: sqadd     z0.b, z0.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqadd     z0.b, z0.b, #1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: sqadd     z0.b, z0.b, #1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqadd     z0.b, z0.b, #256
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: sqadd     z0.b, z0.b, #256
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqadd     z0.h, z0.h, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sqadd     z0.h, z0.h, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqadd     z0.h, z0.h, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sqadd     z0.h, z0.h, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqadd     z0.h, z0.h, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sqadd     z0.h, z0.h, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqadd     z0.s, z0.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sqadd     z0.s, z0.s, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqadd     z0.s, z0.s, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sqadd     z0.s, z0.s, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqadd     z0.s, z0.s, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sqadd     z0.s, z0.s, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqadd     z0.d, z0.d, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sqadd     z0.d, z0.d, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqadd     z0.d, z0.d, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sqadd     z0.d, z0.d, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqadd     z0.d, z0.d, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sqadd     z0.d, z0.d, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
