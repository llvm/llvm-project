// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

// SVE2p2 should imply SVE2p1
.cpu generic+sve2p2
sclamp z0.s, z1.s, z2.s
// CHECK: sclamp z0.s, z1.s, z2.s

.cpu generic+sve2p2
fcvtnt  z0.s, p0/z, z0.d
// CHECK: fcvtnt  z0.s, p0/z, z0.d