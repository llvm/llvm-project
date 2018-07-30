// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element size

udot  z0.s, z1.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot  z0.s, z1.h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot  z0.d, z1.b, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot  z0.d, z1.b, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot  z0.d, z1.s, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot  z0.d, z1.s, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid restricted register for indexed vector.

udot  z0.s, z1.b, z8.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: udot  z0.s, z1.b, z8.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot  z0.d, z1.h, z16.h[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: udot  z0.d, z1.h, z16.h[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element index

udot  z0.s, z1.b, z7.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: udot  z0.s, z1.b, z7.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot  z0.s, z1.b, z7.b[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: udot  z0.s, z1.b, z7.b[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot  z0.d, z1.h, z15.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: udot  z0.d, z1.h, z15.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot  z0.d, z1.h, z15.h[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: udot  z0.d, z1.h, z15.h[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
udot  z0.d, z1.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: udot  z0.d, z1.h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.d, p0/z, z7.d
udot  z0.d, z1.h, z15.h[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: udot  z0.d, z1.h, z15.h[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
