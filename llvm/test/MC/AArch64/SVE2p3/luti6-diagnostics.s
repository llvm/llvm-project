// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid element width

luti6 z10.h, { z0.b, z1.b }, z0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti6 z10.h, { z0.b, z1.b }, z0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 z10.s, { z0.b, z1.b }, z0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: luti6 z10.s, { z0.b, z1.b }, z0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.h, p0/m, z7.h
luti6 z0.b, { z2.b, z3.b }, z4
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 z0.b, { z2.b, z3.b }, z4
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
luti6 z0.b, { z2.b, z3.b }, z4
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 z0.b, { z2.b, z3.b }, z4
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element width

luti6 z10.s, { z0.h, z1.h }, z0[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: luti6 z10.s, { z0.h, z1.h }, z0[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 z10.b, { z0.h, z1.h }, z0[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti6 z10.b, { z0.h, z1.h }, z0[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate range

luti6 z10.h, { z0.h, z1.h }, z0[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: luti6 z10.h, { z0.h, z1.h }, z0[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 z10.h, { z0.h, z1.h }, z0[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: luti6 z10.h, { z0.h, z1.h }, z0[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.h, p0/m, z7.h
luti6 z0.h, { z2.h, z3.h }, z4[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 z0.h, { z2.h, z3.h }, z4[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
luti6 z0.h, { z2.h, z3.h }, z4[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 z0.h, { z2.h, z3.h }, z4[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
