// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid operand for instruction

fcvtzsn z0.b, { z0.b, z1.b }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtzsn z0.b, { z0.b, z1.b }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtzsn z0.h, { z0.h, z1.h }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtzsn z0.h, { z0.h, z1.h }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtzsn z0.s, { z0.s, z1.s }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtzsn z0.s, { z0.s, z1.s }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtzsn z0.b, { z1.h, z2.h }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: fcvtzsn z0.b, { z1.h, z2.h }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
fcvtzsn z0.b, { z2.h, z3.h }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcvtzsn z0.b, { z2.h, z3.h }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid operand for instruction

fcvtzun z0.b, { z0.b, z1.b }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtzun z0.b, { z0.b, z1.b }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtzun z0.h, { z0.h, z1.h }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtzun z0.h, { z0.h, z1.h }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtzun z0.s, { z0.s, z1.s }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtzun z0.s, { z0.s, z1.s }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtzun z0.b, { z1.h, z2.h }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: fcvtzun z0.b, { z1.h, z2.h }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
fcvtzun z0.b, { z2.h, z3.h }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcvtzun z0.b, { z2.h, z3.h }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element width

scvtf z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtf z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtf z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtf z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtf z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtf z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtf z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtf z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
scvtf z0.h, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: scvtf z0.h, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element width

scvtflt z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtflt z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtflt z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtflt z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtflt z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtflt z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

scvtflt z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: scvtflt z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
scvtflt z0.h, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: scvtflt z0.h, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element width

ucvtf z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtf z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ucvtf z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtf z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ucvtf z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtf z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ucvtf z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtf z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
ucvtf z0.h, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ucvtf z0.h, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid element width

ucvtflt z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtflt z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ucvtflt z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtflt z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ucvtflt z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtflt z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ucvtflt z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ucvtflt z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
ucvtflt z0.h, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: ucvtflt z0.h, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
