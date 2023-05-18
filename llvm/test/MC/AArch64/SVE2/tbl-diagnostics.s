// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s

tbl z0.b, { z1.b, z2.b }, z3.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: tbl z0.b, { z1.b, z2.b }, z3.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

tbl z0.d, { }, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: tbl z0.d, { }, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

tbl z0.d, { z1.d, z2.d, z3.d }, z4.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: tbl z0.d, { z1.d, z2.d, z3.d }, z4.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

tbl z0.d, { z1.d, z2.b }, z3.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: tbl z0.d, { z1.d, z2.b }, z3.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

tbl z0.d, { z1.d, z21.d }, z3.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: tbl z0.d, { z1.d, z21.d }, z3.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

tbl z0.d, { v0.2d, v1.2d }, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: tbl z0.d, { v0.2d, v1.2d }, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
tbl  z31.d, { z30.d, z31.d }, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: tbl  z31.d, { z30.d, z31.d }, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
tbl  z31.d, { z30.d, z31.d }, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: tbl  z31.d, { z30.d, z31.d }, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
