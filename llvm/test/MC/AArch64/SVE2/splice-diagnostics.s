// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element widths.

splice z0.b, p0, { z1.h, z2.h }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: splice z0.b, p0, { z1.h, z2.h }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

splice z0.b, p0, { }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: splice z0.b, p0, { }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

splice z0.b, p0, { z1.b }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: splice z0.b, p0, { z1.b }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

splice z0.b, p0, { z1.b, z2.b, z3.b }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: splice z0.b, p0, { z1.b, z2.b, z3.b }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

splice z0.b, p0, { z1.b, z2.h }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: splice z0.b, p0, { z1.b, z2.h }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

splice z0.b, p0, { z1.b, z31.b }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must be sequential
// CHECK-NEXT: splice z0.b, p0, { z1.b, z31.b }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

splice z0.b, p0, { v0.4b, v1.4b }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: splice z0.b, p0, { v0.4b, v1.4b }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid predicate operation

splice z0.b, p0/z, { z1.b, z2.b }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: splice z0.b, p0/z, { z1.b, z2.b }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

splice z0.b, p0/m, { z1.b, z2.b }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: splice z0.b, p0/m, { z1.b, z2.b }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Predicate not in restricted predicate range

splice z0.b, p8, { z1.b, z2.b }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: splice z0.b, p8, { z1.b, z2.b }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31, z6
splice z31.b, p0, { z30.b, z31.b }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: splice z31.b, p0, { z30.b, z31.b }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31.b, p0/z, z6.b
splice z31.b, p0, { z30.b, z31.b }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: splice z31.b, p0, { z30.b, z31.b }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
