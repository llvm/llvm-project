// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s

srsra z30.b, z10.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8]
// CHECK-NEXT: srsra z30.b, z10.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

srsra z18.b, z27.b, #9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8]
// CHECK-NEXT: srsra z18.b, z27.b, #9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

srsra z26.h, z4.h, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: srsra z26.h, z4.h, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

srsra z25.h, z10.h, #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: srsra z25.h, z10.h, #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

srsra z17.s, z0.s, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32]
// CHECK-NEXT: srsra z17.s, z0.s, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

srsra z0.s, z15.s, #33
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32]
// CHECK-NEXT: srsra z0.s, z15.s, #33
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

srsra z4.d, z13.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]
// CHECK-NEXT: srsra z4.d, z13.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

srsra z26.d, z26.d, #65
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]
// CHECK-NEXT: srsra z26.d, z26.d, #65
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Element sizes must match

srsra z0.b, z0.d, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: srsra z0.b, z0.d, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
srsra     z0.d, z1.d, #64
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: srsra     z0.d, z1.d, #64
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
