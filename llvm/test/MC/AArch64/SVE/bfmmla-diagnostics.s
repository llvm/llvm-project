// RUN: not llvm-mc -triple=aarch64 -mattr=+sve,bf16  2>&1 < %s| FileCheck %s

bfmmla z0.s, z1.s, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmmla z0.s, z1.s, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmmla z0.h, z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmmla z0.h, z1.h, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmmla z0.s, z1.h, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmmla z0.s, z1.h, z2.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.s, p0/m, z7.s
bfmmla z0.s, z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: bfmmla z0.s, z1.h, z2.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
