// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Register z32 does not exist.
add z22.h, z10.h, z32.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: add z22.h, z10.h, z32.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Invalid element kind.
add z20.h, z2.h, z31.x
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: add z20.h, z2.h, z31.x
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Element size specifiers should match.
add z27.h, z11.h, z27.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: add z27.h, z11.h, z27.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Invalid predicate suffix '/a'
add z29.d, p7/a, z29.d, z8.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expecting 'm' or 'z' predication
// CHECK-NEXT: add z29.d, p7/a, z29.d, z8.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Missing predicate suffix
add z29.d, p7, z29.d, z8.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: add z29.d, p7, z29.d, z8.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// error: restricted predicate has range [0, 7].

add z22.b, p8/m, z22.b, z11.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: add z22.b, p8/m, z22.b, z11.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add z22.h, p8/m, z22.h, z6.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: add z22.h, p8/m, z22.h, z6.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add z30.s, p8/m, z30.s, z13.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: add z30.s, p8/m, z30.s, z13.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add z29.d, p8/m, z29.d, z8.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: add z29.d, p8/m, z29.d, z8.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Source and Destination Registers must match

add z19.b, p4/m, z20.b, z13.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: add z19.b, p4/m, z20.b, z13.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add z9.h, p3/m, z10.h, z28.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: add z9.h, p3/m, z10.h, z28.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add z5.s, p3/m, z6.s, z18.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: add z5.s, p3/m, z6.s, z18.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add z9.d, p4/m, z10.d, z7.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: add z9.d, p4/m, z10.d, z7.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediates

add     z0.b, z0.b, #0, lsl #8      // #0, lsl #8 is not valid for .b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: add     z0.b, z0.b, #0, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     z0.b, z0.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: add     z0.b, z0.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     z0.b, z0.b, #1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: add     z0.b, z0.b, #1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     z0.b, z0.b, #256
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: add     z0.b, z0.b, #256
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     z0.h, z0.h, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: add     z0.h, z0.h, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     z0.h, z0.h, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: add     z0.h, z0.h, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     z0.h, z0.h, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: add     z0.h, z0.h, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     z0.s, z0.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: add     z0.s, z0.s, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     z0.s, z0.s, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: add     z0.s, z0.s, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     z0.s, z0.s, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: add     z0.s, z0.s, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     z0.d, z0.d, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: add     z0.d, z0.d, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     z0.d, z0.d, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: add     z0.d, z0.d, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

add     z0.d, z0.d, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: add     z0.d, z0.d, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
add     z31.d, z31.d, #65280
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: add     z31.d, z31.d, #65280
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z23.s, p0/z, z30.s
add     z23.s, z13.s, z8.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: add     z23.s, z13.s, z8.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z23, z30
add     z23.s, z13.s, z8.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: add     z23.s, z13.s, z8.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
