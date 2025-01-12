// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

// SME2p2 should imply SME2p1
.cpu generic+sme2p2
movaz {z0.d, z1.d}, za.d[w8, 0, vgx2]
// CHECK: movaz { z0.d, z1.d }, za.d[w8, 0, vgx2]

.cpu generic+sme2p2
ftmopa  za0.s, {z0.s-z1.s}, z0.s, z20[0]
// CHECK: ftmopa za0.s, { z0.s, z1.s }, z0.s, z20[0]