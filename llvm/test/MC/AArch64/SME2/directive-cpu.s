// RUN: llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

// SME2 should imply SME
.cpu generic+sme2
zero {za}
// CHECK: zero {za}

.cpu generic+sme2
sqcvt z0.h, {z0.s, z1.s}
// CHECK: sqcvt z0.h, { z0.s, z1.s }

.cpu generic+sme2+sve-b16b16
bfclamp {z0.h, z1.h}, z0.h, z0.h
// CHECK: bfclamp { z0.h, z1.h }, z0.h, z0.h

.cpu generic+sme-b16b16
bfadd za.h[w8, 0, vgx2], {z0.h, z1.h}
// CHECK: bfadd   za.h[w8, 0, vgx2], { z0.h, z1.h }
