// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+ssve-fp8fma  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// z register out of range for index

fmlallbb z0.s, z1.b, z8.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlallbb z0.s, z1.b, z8.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlallbt z0.s, z1.b, z8.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlallbt z0.s, z1.b, z8.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalltb z0.s, z1.b, z8.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlalltb z0.s, z1.b, z8.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalltt z0.s, z1.b, z8.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlalltt z0.s, z1.b, z8.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Index out of bounds

fmlallbb z0.s, z1.b, z7.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlallbb z0.s, z1.b, z7.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlallbt z0.s, z1.b, z7.b[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlallbt z0.s, z1.b, z7.b[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalltb z0.s, z1.b, z7.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlalltb z0.s, z1.b, z7.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalltt z0.s, z1.b, z7.b[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlalltt z0.s, z1.b, z7.b[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid element width

fmlallbb z0.h, z1.b, z2.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlallbb z0.h, z1.b, z2.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlallbt z0.h, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlallbt  z0.h, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalltb z0.s, z1.h, z2.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlalltb z0.s, z1.h, z2.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalltt z0.s, z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmlalltt  z0.s, z1.h, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.s, p0/z, z0.s
fmlallbb  z0.s, z1.b, z7.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: fmlallbb  z0.s, z1.b, z7.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z29.s, p0/z, z7.s
fmlalltt  z29.s, z30.b, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: fmlalltt  z29.s, z30.b, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
