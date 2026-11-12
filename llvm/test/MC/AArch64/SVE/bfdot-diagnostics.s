// RUN: not llvm-mc -triple=aarch64 -mattr=+sve,bf16  2>&1 < %s| FileCheck %s

bfdot z0.s, z1.s, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfdot z0.s, z1.s, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot z0.h, z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfdot z0.h, z1.h, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot z0.s, z1.h, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z7.h
// CHECK-NEXT: bfdot z0.s, z1.h, z2.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.s, p0/m, z7.s
bfdot z0.s, z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: bfdot z0.s, z1.h, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot z0.s, z1.s, z2.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfdot z0.s, z1.s, z2.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot z0.h, z1.h, z2.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfdot z0.h, z1.h, z2.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot z0.s, z1.h, z2.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z7.h
// CHECK-NEXT: bfdot z0.s, z1.h, z2.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot z0.s, z1.h, z8.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfdot z0.s, z1.h, z8.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfdot z0.s, z1.h, z2.h[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: bfdot z0.s, z1.h, z2.h[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.s, p0/m, z7.s
bfdot z0.s, z1.h, z2.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: bfdot z0.s, z1.h, z2.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
