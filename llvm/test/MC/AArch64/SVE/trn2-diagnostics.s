// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Invalid element kind.
trn2 z6.h, z23.h, z31.x
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: trn2 z6.h, z23.h, z31.x
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Element size specifiers should match.
trn2 z0.h, z30.h, z24.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: trn2 z0.h, z30.h, z24.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Too few operands
trn2 z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// CHECK-NEXT: trn2 z1.h, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// z32 is not a valid SVE data register
trn2 z1.s, z2.s, z32.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: trn2 z1.s, z2.s, z32.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// p16 is not a valid SVE predicate register
trn2 p1.s, p2.s, p16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: trn2 p1.s, p2.s, p16.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Combining data and predicate registers as operands
trn2 z1.s, z2.s, p3.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: trn2 z1.s, z2.s, p3.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Combining predicate and data registers as operands
trn2 p1.s, p2.s, z3.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: trn2 p1.s, p2.s, z3.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
trn2    z31.d, z31.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: trn2    z31.d, z31.d, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
trn2    z31.d, z31.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: trn2    z31.d, z31.d, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
