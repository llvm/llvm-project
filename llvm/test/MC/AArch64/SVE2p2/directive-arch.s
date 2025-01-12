// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

// SVE2p2 should imply SVE2p1
.arch armv9-a+sve2p1
sclamp z0.s, z1.s, z2.s
// CHECK: sclamp z0.s, z1.s, z2.s

.arch armv9-a+sve2p2
bfcvtnt z23.h, p3/z, z13.s
// CHECK: bfcvtnt z23.h, p3/z, z13.s