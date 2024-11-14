// RUN: not llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

// SME2 should require SME
.cpu generic+sme2+nosme
sqcvt z0.h, { z0.s, z1.s }
// CHECK: error: instruction requires: sme2
// CHECK: sqcvt z0.h, { z0.s, z1.s }

.cpu generic+sme2+nosme2
sqcvt z0.h, { z0.s, z1.s }
// CHECK: error: instruction requires: sme2
// CHECK: sqcvt z0.h, { z0.s, z1.s }

.cpu generic+sme2+sve-b16b16+nosve-b16b16
bfclamp {z0.h, z1.h}, z0.h, z0.h
// CHECK: error: instruction requires: sve-b16b16
// CHECK: bfclamp {z0.h, z1.h}, z0.h, z0.h

.cpu generic+sme-b16b16+nosme-b16b16
bfadd za.h[w8, 0, vgx2], {z0.h, z1.h}
// CHECK: error: instruction requires: sme-b16b16
// CHECK: bfadd za.h[w8, 0, vgx2], {z0.h, z1.h}