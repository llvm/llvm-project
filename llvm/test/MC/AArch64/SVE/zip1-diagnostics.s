// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Invalid element kind.
zip1 z10.h, z22.h, z31.x
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: zip1 z10.h, z22.h, z31.x
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Element size specifiers should match.
zip1 z10.h, z3.h, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: zip1 z10.h, z3.h, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Too few operands
zip1 z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// CHECK-NEXT: zip1 z1.h, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// z32 is not a valid SVE data register
zip1 z1.s, z2.s, z32.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: zip1 z1.s, z2.s, z32.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// p16 is not a valid SVE predicate register
zip1 p1.s, p2.s, p16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: zip1 p1.s, p2.s, p16.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Combining data and predicate registers as operands
zip1 z1.s, z2.s, p3.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: zip1 z1.s, z2.s, p3.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Combining predicate and data registers as operands
zip1 p1.s, p2.s, z3.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: zip1 p1.s, p2.s, z3.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
