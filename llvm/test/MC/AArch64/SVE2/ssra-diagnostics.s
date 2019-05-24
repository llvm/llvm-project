// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2  2>&1 < %s| FileCheck %s

ssra z30.b, z10.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8]
// CHECK-NEXT: ssra z30.b, z10.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ssra z18.b, z27.b, #9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 8]
// CHECK-NEXT: ssra z18.b, z27.b, #9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ssra z26.h, z4.h, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: ssra z26.h, z4.h, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ssra z25.h, z10.h, #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: ssra z25.h, z10.h, #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ssra z17.s, z0.s, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32]
// CHECK-NEXT: ssra z17.s, z0.s, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ssra z0.s, z15.s, #33
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 32]
// CHECK-NEXT: ssra z0.s, z15.s, #33
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ssra z4.d, z13.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]
// CHECK-NEXT: ssra z4.d, z13.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ssra z26.d, z26.d, #65
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 64]
// CHECK-NEXT: ssra z26.d, z26.d, #65
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Element sizes must match

ssra z0.b, z0.d, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ssra z0.b, z0.d, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
ssra     z0.d, z1.d, #64
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: ssra     z0.d, z1.d, #64
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
