// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

// SVE2p2 should imply SVE2p1
.arch_extension sve2p2
sclamp z0.s, z1.s, z2.s
// CHECK: sclamp z0.s, z1.s, z2.s

.arch_extension sve2p2
bfcvtnt z0.h, p0/z, z0.s
// CHECK: bfcvtnt z0.h, p0/z, z0.s