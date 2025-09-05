// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid element width

sqrshrn z10.s, { z0.s, z1.s }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqrshrn z10.s, { z0.s, z1.s }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrshrn z10.d, { z0.d, z1.d }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqrshrn z10.d, { z0.d, z1.d }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid operand for instruction

sqrshrn z10.h, { z0.b, z1.b }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqrshrn z10.h, { z0.b, z1.b }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Intermediate out of range

sqrshrn z10.b, { z0.h, z1.h }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8].
// CHECK-NEXT: sqrshrn z10.b, { z0.h, z1.h }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrshrn z10.b, { z0.h, z1.h }, #9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8].
// CHECK-NEXT: sqrshrn z10.b, { z0.h, z1.h }, #9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
sqrshrn z10.b, { z0.h, z1.h }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqrshrn z10.b, { z0.h, z1.h }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element width

sqrshrun z10.s, { z0.s, z1.s }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqrshrun z10.s, { z0.s, z1.s }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrshrun z10.d, { z0.d, z1.d }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqrshrun z10.d, { z0.d, z1.d }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid operand for instruction

sqrshrun z10.h, { z0.b, z1.b }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqrshrun z10.h, { z0.b, z1.b }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Intermediate out of range

sqrshrun z10.b, { z0.h, z1.h }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8].
// CHECK-NEXT: sqrshrun z10.b, { z0.h, z1.h }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqrshrun z10.b, { z0.h, z1.h }, #9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8].
// CHECK-NEXT: sqrshrun z10.b, { z0.h, z1.h }, #9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
sqrshrun z10.b, { z0.h, z1.h }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqrshrun z10.b, { z0.h, z1.h }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element width

sqshrn z10.s, { z0.s, z1.s }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqshrn z10.s, { z0.s, z1.s }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrn z10.d, { z0.d, z1.d }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqshrn z10.d, { z0.d, z1.d }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Intermediate out of range

sqshrn z10.b, { z0.h, z1.h }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8].
// CHECK-NEXT: sqshrn z10.b, { z0.h, z1.h }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrn z10.b, { z0.h, z1.h }, #9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8].
// CHECK-NEXT: sqshrn z10.b, { z0.h, z1.h }, #9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrn z0.h, { z0.s, z1.s }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16].
// CHECK-NEXT: sqshrn z0.h, { z0.s, z1.s }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrn z0.h, { z0.s, z1.s }, #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16].
// CHECK-NEXT: sqshrn z0.h, { z0.s, z1.s }, #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
sqshrn z10.b, { z0.h, z1.h }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqshrn z10.b, { z0.h, z1.h }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element width

sqshrun z10.s, { z0.s, z1.s }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqshrun z10.s, { z0.s, z1.s }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrun z10.d, { z0.d, z1.d }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqshrun z10.d, { z0.d, z1.d }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Intermediate out of range

sqshrun z10.b, { z0.h, z1.h }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8].
// CHECK-NEXT: sqshrun z10.b, { z0.h, z1.h }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrun z10.b, { z0.h, z1.h }, #9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8].
// CHECK-NEXT: sqshrun z10.b, { z0.h, z1.h }, #9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrun z10.h, { z0.s, z1.s }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16].
// CHECK-NEXT: sqshrun z10.h, { z0.s, z1.s }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshrun z10.h, { z0.s, z1.s }, #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16].
// CHECK-NEXT: sqshrun z10.h, { z0.s, z1.s }, #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
sqshrun z10.b, { z0.h, z1.h }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sqshrun z10.b, { z0.h, z1.h }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element width

uqrshrn z10.s, { z0.s, z1.s }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: uqrshrn z10.s, { z0.s, z1.s }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqrshrn z10.d, { z0.d, z1.d }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: uqrshrn z10.d, { z0.d, z1.d }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid operand for instruction

uqrshrn z10.h, { z0.b, z1.b }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: uqrshrn z10.h, { z0.b, z1.b }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Intermediate out of range

uqrshrn z10.b, { z0.h, z1.h }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8].
// CHECK-NEXT: uqrshrn z10.b, { z0.h, z1.h }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqrshrn z10.b, { z0.h, z1.h }, #9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8].
// CHECK-NEXT: uqrshrn z10.b, { z0.h, z1.h }, #9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
uqrshrn z10.b, { z0.h, z1.h }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: uqrshrn z10.b, { z0.h, z1.h }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element width

uqshrn z10.s, { z0.s, z1.s }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: uqshrn z10.s, { z0.s, z1.s }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqshrn z10.d, { z0.d, z1.d }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: uqshrn z10.d, { z0.d, z1.d }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Intermediate out of range

uqshrn z10.b, { z0.h, z1.h }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8].
// CHECK-NEXT: uqshrn z10.b, { z0.h, z1.h }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqshrn z10.b, { z0.h, z1.h }, #9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8].
// CHECK-NEXT: uqshrn z10.b, { z0.h, z1.h }, #9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqshrn z0.h, { z0.s, z1.s }, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16].
// CHECK-NEXT: uqshrn z0.h, { z0.s, z1.s }, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqshrn z0.h, { z0.s, z1.s }, #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16].
// CHECK-NEXT: uqshrn z0.h, { z0.s, z1.s }, #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
uqshrn z10.b, { z0.h, z1.h }, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: uqshrn z10.b, { z0.h, z1.h }, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
