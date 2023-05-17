// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sve2p1
.arch_extension nosve2p1
sclamp z0.s, z1.s, z2.s
// CHECK: error: instruction requires: sme or sve2p1
// CHECK: sclamp z0.s, z1.s, z2.s
