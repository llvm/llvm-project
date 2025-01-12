// RUN: llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

// SME2p1 should imply SME2
.cpu generic+sme2p1
sqcvt z0.h, {z0.s, z1.s}
// CHECK: sqcvt z0.h, { z0.s, z1.s }

.cpu generic+sme2p1
movaz   {z0.d, z1.d}, za.d[w8, 0, vgx2]
// CHECK: movaz { z0.d, z1.d }, za.d[w8, 0, vgx2]

.cpu generic+sve-aes2+ssve-aes
aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.q[0]
// CHECK: aesdimc { z0.b - z3.b }, { z0.b - z3.b }, z0.q[0]