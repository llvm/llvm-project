// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sve2p2
bfcvtnt z0.h, p0/z, z0.s
// CHECK: bfcvtnt z0.h, p0/z, z0.s