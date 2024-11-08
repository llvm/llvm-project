// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sve2p2
.arch_extension nosve2p2
bfcvtnt z0.h, p0/z, z0.s
// CHECK: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: bfcvtnt z0.h, p0/z, z0.s