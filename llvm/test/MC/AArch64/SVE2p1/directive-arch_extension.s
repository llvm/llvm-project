// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sve2p1
sclamp z0.s, z1.s, z2.s
// CHECK: sclamp z0.s, z1.s, z2.s

.arch_extension sve2p1
.arch_extension sve-b16b16
bfadd   z23.h, p3/m, z23.h, z13.h
// CHECK: bfadd   z23.h, p3/m, z23.h, z13.h