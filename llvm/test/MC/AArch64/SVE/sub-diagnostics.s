// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Register z32 does not exist.
sub z3.h, z26.h, z32.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sub z3.h, z26.h, z32.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Invalid element kind.
sub z4.h, z27.h, z31.x
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: sub z4.h, z27.h, z31.x
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Element size specifiers should match.
sub z0.h, z8.h, z8.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sub z0.h, z8.h, z8.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Invalid predicate suffix '/a'
sub z29.d, p7/a, z29.d, z8.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expecting 'm' or 'z' predication
// CHECK-NEXT: sub z29.d, p7/a, z29.d, z8.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Missing predicate suffix
sub z29.d, p7, z29.d, z8.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sub z29.d, p7, z29.d, z8.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// error: restricted predicate has range [0, 7].

sub z26.b, p8/m, z26.b, z27.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: sub z26.b, p8/m, z26.b, z27.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub z14.h, p8/m, z14.h, z18.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: sub z14.h, p8/m, z14.h, z18.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub z30.s, p8/m, z30.s, z23.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: sub z30.s, p8/m, z30.s, z23.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub z29.d, p8/m, z29.d, z3.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: sub z29.d, p8/m, z29.d, z3.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Source and Destination Registers must match

sub z25.b, p4/m, z26.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: sub z25.b, p4/m, z26.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub z29.h, p6/m, z30.h, z20.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: sub z29.h, p6/m, z30.h, z20.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub z14.s, p2/m, z15.s, z21.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: sub z14.s, p2/m, z15.s, z21.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub z2.d, p5/m, z3.d, z11.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: sub z2.d, p5/m, z3.d, z11.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediates

sub     z0.b, z0.b, #0, lsl #8      // #0, lsl #8 is not valid for .b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: sub     z0.b, z0.b, #0, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub     z0.b, z0.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: sub     z0.b, z0.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub     z0.b, z0.b, #1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: sub     z0.b, z0.b, #1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub     z0.b, z0.b, #256
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: sub     z0.b, z0.b, #256
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub     z0.h, z0.h, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sub     z0.h, z0.h, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub     z0.h, z0.h, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sub     z0.h, z0.h, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub     z0.h, z0.h, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sub     z0.h, z0.h, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub     z0.s, z0.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sub     z0.s, z0.s, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub     z0.s, z0.s, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sub     z0.s, z0.s, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub     z0.s, z0.s, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sub     z0.s, z0.s, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub     z0.d, z0.d, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sub     z0.d, z0.d, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub     z0.d, z0.d, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sub     z0.d, z0.d, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub     z0.d, z0.d, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: sub     z0.d, z0.d, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
sub     z31.d, z31.d, #65280
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: sub     z31.d, z31.d, #65280
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31.s, p0/z, z6.s
sub     z31.s, z31.s, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sub     z31.s, z31.s, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
sub     z31.s, z31.s, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sub     z31.s, z31.s, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
