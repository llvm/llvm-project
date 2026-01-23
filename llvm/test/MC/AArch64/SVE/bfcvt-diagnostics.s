// RUN: not llvm-mc -triple=aarch64 -mattr=+sve,bf16 2>&1 < %s| FileCheck %s

bfcvt z0.s, p0/m, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfcvt z0.s, p0/m, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfcvt z0.h, p0/m, z1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfcvt z0.h, p0/m, z1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfcvt z0.h, p0/z, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: bfcvt z0.h, p0/z, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfcvt z0.h, p8/m, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: bfcvt z0.h, p8/m, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.h, p0/m, z7.h
bfcvt z0.h, p0/m, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx with a different element size
// CHECK-NEXT: bfcvt z0.h, p0/m, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
