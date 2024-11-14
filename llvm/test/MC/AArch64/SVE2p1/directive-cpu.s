// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

// SVE2p1 should imply SVE2
.cpu generic+sve2p1
tbx z0.b, z1.b, z2.b
// CHECK: tbx z0.b, z1.b, z2.b

.cpu generic+sve2p1
sclamp z0.s, z1.s, z2.s
// CHECK: sclamp z0.s, z1.s, z2.s

.cpu generic+sve2p1+sve-b16b16
bfadd   z23.h, p3/m, z23.h, z13.h
// CHECK: bfadd   z23.h, p3/m, z23.h, z13.h

.cpu generic+sve2p1+sve-aes2
aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.q[0]
// CHECK: aesdimc { z0.b - z3.b }, { z0.b - z3.b }, z0.q[0]