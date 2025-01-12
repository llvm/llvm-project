// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

// SME2p1 should require SME2
.arch_extension sme2p1
.arch_extension nosme2
movaz {z0.d, z1.d}, za.d[w8, 0]
// CHECK: error: instruction requires: sme2p1
// CHECK-NEXT: movaz {z0.d, z1.d}, za.d[w8, 0]

.arch_extension sve-aes2
.arch_extension ssve-aes
.arch_extension nossve-aes
aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.q[0]
// CHECK: error: instruction requires: sve2p1 or ssve-aes
// CHECK-NEXT: aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.q[0]