// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

uqdecd wsp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecd wsp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecd sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecd sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecd z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: uqdecd z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Operands not matching up (unsigned dec only has one register operand)

uqdecd x0, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecd x0, w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecd w0, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecd w0, w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecd x0, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecd x0, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Immediate not compatible with encode/decode function.

uqdecd x0, all, mul #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: uqdecd x0, all, mul #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecd x0, all, mul #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: uqdecd x0, all, mul #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecd x0, all, mul #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: uqdecd x0, all, mul #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate patterns

uqdecd x0, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecd x0, vl512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecd x0, vl9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: uqdecd x0, vl9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecd x0, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: uqdecd x0, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecd x0, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: uqdecd x0, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
uqdecd  z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: uqdecd  z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.d, p0/z, z7.d
uqdecd  z0.d, pow2, mul #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: uqdecd  z0.d, pow2, mul #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.d, p0/z, z7.d
uqdecd  z0.d, pow2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: uqdecd  z0.d, pow2
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
