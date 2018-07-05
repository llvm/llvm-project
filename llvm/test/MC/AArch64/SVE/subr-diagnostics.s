// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Invalid element kind.
subr z0.h, p0/m, z0.h, z0.x
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: subr z0.h, p0/m, z0.h, z0.x
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Element size specifiers should match.
subr z0.h, p0/m, z0.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: subr z0.h, p0/m, z0.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Invalid predicate suffix '/a'
subr z0.d, p7/a, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expecting 'm' or 'z' predication
// CHECK-NEXT: subr z0.d, p7/a, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Missing predicate suffix
subr z0.d, p7, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: subr z0.d, p7, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// error: restricted predicate has range [0, 7].

subr z26.b, p8/m, z26.b, z27.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: subr z26.b, p8/m, z26.b, z27.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr z14.h, p8/m, z14.h, z18.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: subr z14.h, p8/m, z14.h, z18.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr z30.s, p8/m, z30.s, z23.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: subr z30.s, p8/m, z30.s, z23.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr z29.d, p8/m, z29.d, z3.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: subr z29.d, p8/m, z29.d, z3.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Source and Destination Registers must match

subr z25.b, p4/m, z26.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: subr z25.b, p4/m, z26.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr z29.h, p6/m, z30.h, z20.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: subr z29.h, p6/m, z30.h, z20.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr z14.s, p2/m, z15.s, z21.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: subr z14.s, p2/m, z15.s, z21.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr z2.d, p5/m, z3.d, z11.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: subr z2.d, p5/m, z3.d, z11.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediates

subr     z0.b, z0.b, #0, lsl #8      // #0, lsl #8 is not valid for .b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: subr     z0.b, z0.b, #0, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr     z0.b, z0.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: subr     z0.b, z0.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr     z0.b, z0.b, #1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: subr     z0.b, z0.b, #1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr     z0.b, z0.b, #256
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] with a shift amount of 0
// CHECK-NEXT: subr     z0.b, z0.b, #256
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr     z0.h, z0.h, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: subr     z0.h, z0.h, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr     z0.h, z0.h, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: subr     z0.h, z0.h, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr     z0.h, z0.h, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: subr     z0.h, z0.h, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr     z0.s, z0.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: subr     z0.s, z0.s, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr     z0.s, z0.s, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: subr     z0.s, z0.s, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr     z0.s, z0.s, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: subr     z0.s, z0.s, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr     z0.d, z0.d, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: subr     z0.d, z0.d, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr     z0.d, z0.d, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: subr     z0.d, z0.d, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

subr     z0.d, z0.d, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 255] or a multiple of 256 in range [256, 65280]
// CHECK-NEXT: subr     z0.d, z0.d, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
