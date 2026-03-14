// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

sqdech w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdech w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdech wsp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdech wsp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdech sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdech sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdech z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqdech z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Operands not matching up

sqdech x0, w1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be 32-bit form of destination register
// CHECK-NEXT: sqdech x0, w1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdech x0, x1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdech x0, x1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Immediate not compatible with encode/decode function.

sqdech x0, all, mul #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqdech x0, all, mul #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdech x0, all, mul #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqdech x0, all, mul #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdech x0, all, mul #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqdech x0, all, mul #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate patterns

sqdech x0, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdech x0, vl512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdech x0, vl9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqdech x0, vl9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdech x0, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: sqdech x0, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqdech x0, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: sqdech x0, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.h, p0/z, z7.h
sqdech  z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: sqdech  z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.h, p0/z, z7.h
sqdech  z0.h, pow2, mul #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: sqdech  z0.h, pow2, mul #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.h, p0/z, z7.h
sqdech  z0.h, pow2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: sqdech  z0.h, pow2
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
