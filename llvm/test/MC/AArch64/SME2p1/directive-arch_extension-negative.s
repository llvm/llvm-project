// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sme2p1
.arch_extension nosme2
sqcvt z0.h, { z0.s, z1.s }
// CHECK: error: instruction requires: sme2
// CHECK: sqcvt z0.h

.arch_extension sme2
.arch_extension sve-b16b16
.arch_extension nosve-b16b16
bfclamp { z0.h, z1.h }, z0.h, z0.h
// CHECK: error: instruction requires: sve-b16b16
// CHECK: bfclamp { z0.h, z1.h }, z0.h, z0.h

.arch_extension sme-b16b16
.arch_extension nosme-b16b16
bfadd za.h[w8, 3], {z20.h-z21.h}
// CHECK: error: instruction requires: sme-b16b16
// CHECK: bfadd za.h[w8, 3], {z20.h-z21.h}

.arch_extension sve-aes2
.arch_extension ssve-aes
.arch_extension nossve-aes
aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.q[0]
// CHECK: error: instruction requires: sve2p1 or ssve-aes
// CHECK: aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.q[0]