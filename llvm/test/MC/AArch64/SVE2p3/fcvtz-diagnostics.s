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

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
fcvtzsn z0.b, { z0.h, z1.h }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcvtzsn z0.b, { z0.h, z1.h }
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

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0, z7
fcvtzun z0.b, { z0.h, z1.h }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fcvtzun z0.b, { z0.h, z1.h }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
