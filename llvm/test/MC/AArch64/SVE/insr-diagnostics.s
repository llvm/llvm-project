// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid scalar operand size.

insr    z31.b, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: insr    z31.b, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

insr    z31.h, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: insr    z31.h, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

insr    z31.s, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: insr    z31.s, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

insr    z31.d, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: insr    z31.d, w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

insr    z31.b, h0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: insr    z31.b, h0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

insr    z31.h, s0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: insr    z31.h, s0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

insr    z31.s, d0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: insr    z31.s, d0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

insr    z31.d, b0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: insr    z31.d, b0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p0/z, z6.d
insr    z31.d, xzr
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: insr    z31.d, xzr
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z4.d, p0/z, z6.d
insr    z4.d, d31
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: insr    z4.d, d31
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
