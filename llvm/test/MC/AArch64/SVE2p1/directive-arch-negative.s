// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch armv9-a+sve2p1
.arch armv9-a+nosve2p1
sclamp z0.s, z1.s, z2.s
// CHECK: error: instruction requires: sme or sve2p1
// CHECK: sclamp z0.s, z1.s, z2.s

.arch armv9-a+sve2p1+sve-b16b16
.arch armv9-a+sve2p1+nosve-b16b16
bfadd   z23.h, p3/m, z23.h, z13.h
// CHECK: error: instruction requires: sve-b16b16
// CHECK: bfadd   z23.h, p3/m, z23.h, z13.h
