// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+ssve-fp8fma  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// z register out of range for index
fmlalb z0.h, z1.b, z8.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlalb z0.h, z1.b, z8.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalt z0.h, z1.b, z8.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlalt z0.h, z1.b, z8.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Index out of bounds

fmlalb z0.h, z1.b, z7.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlalb z0.h, z1.b, z7.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalt z0.h, z1.b, z7.b[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlalt z0.h, z1.b, z7.b[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid element width

fmlalb z0.h, z1.h, z2.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlalb z0.h, z1.h, z2.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalb z0.h, z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlalb  z0.h, z1.h, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalt z0.s, z1.b, z2.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlalt z0.s, z1.b, z2.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalt z0.s, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlalt  z0.s, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.b, p0/z, z0.b
fmlalb  z0.h, z1.b, z7.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: fmlalb  z0.h, z1.b, z7.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z29.b, p0/z, z7.b
fmlalt  z29.h, z30.b, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: fmlalt  z29.h, z30.b, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
