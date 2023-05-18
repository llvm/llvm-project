// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Invalid result type.

stnt1d { z0.b }, p0, [z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stnt1d { z0.b }, p0, [z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1d { z0.h }, p0, [z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stnt1d { z0.h }, p0, [z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1d { z0.s }, p0, [z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stnt1d { z0.s }, p0, [z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid base vector.

stnt1d { z0.d }, p0, [z0.b]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: stnt1d { z0.d }, p0, [z0.b]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid offset type.

stnt1d { z0.d }, p0, [z0.d, z1.d]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stnt1d { z0.d }, p0, [z0.d, z1.d]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

stnt1d { z27.d }, p8, [z0.d]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: stnt1d { z27.d }, p8, [z0.d]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

stnt1d { }, p0, [z0.d]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: stnt1d { }, p0, [z0.d]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1d { z0.d, z1.d }, p0, [z0.d]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stnt1d { z0.d, z1.d }, p0, [z0.d]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1d { v0.2d }, p0, [z0.d]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stnt1d { v0.2d }, p0, [z0.d]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
stnt1d  { z0.d }, p0, [z0.d, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: stnt1d  { z0.d }, p0, [z0.d, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
stnt1d  { z0.d }, p0, [z0.d, x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: stnt1d  { z0.d }, p0, [z0.d, x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
