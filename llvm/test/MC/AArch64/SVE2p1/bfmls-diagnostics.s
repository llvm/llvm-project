// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1,+b16b16 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector lane index

bfmls z0.h, z0.h, z0.h[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: bfmls z0.h, z0.h, z0.h[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmls z0.h, z0.h, z0.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: bfmls z0.h, z0.h, z0.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmls z0.h, z0.h, z8.h[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z7.h
// CHECK-NEXT: bfmls z0.h, z0.h, z8.h[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

bfmls z0.h, z0.s, z0.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmls z0.h, z0.s, z0.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmls z23.s, z23.h, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmls z23.s, z23.h, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid use of movprfx

movprfx z23.h, p1/m, z31.h
bfmls z23.h, z12.h, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: bfmls z23.h, z12.h, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
