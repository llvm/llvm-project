// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element size

ftmad z0.b, z0.b, z1.b, #7
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ftmad z0.b, z0.b, z1.b, #7
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmad z0.b, z0.b, z1.h, #7
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ftmad z0.b, z0.b, z1.h, #7
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Tied operands must match

ftmad z0.h, z1.h, z2.h, #7
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: ftmad z0.h, z1.h, z2.h, #7
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid immediate range

ftmad z0.h, z0.h, z1.h, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: ftmad z0.h, z0.h, z1.h, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmad z0.h, z0.h, z1.h, #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7].
// CHECK-NEXT: ftmad z0.h, z0.h, z1.h, #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
ftmad z0.d, z0.d, z31.d, #7
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: ftmad z0.d, z0.d, z31.d, #7
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
