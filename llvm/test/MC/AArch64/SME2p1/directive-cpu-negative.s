// RUN: not llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

// SME2p1 should require SME2
.cpu generic+sme2p1+nosme2
movaz {z0.d, z1.d}, za.d[w8, 0]
// CHECK: error: instruction requires: sme2p1
// CHECK-NEXT: movaz {z0.d, z1.d}, za.d[w8, 0]

.cpu generic+sme2p1+nosme2p1
movaz   {z0.d, z1.d}, za.d[w8, 0, vgx2]
// CHECK: error: instruction requires: sme2p1
// CHECK-NEXT: movaz {z0.d, z1.d}, za.d[w8, 0, vgx2]

.cpu generic+sve-aes2+ssve-aes+nossve-aes
aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.q[0]
// CHECK: error: instruction requires: sve2p1 or ssve-aes
// CHECK-NEXT: aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.q[0]