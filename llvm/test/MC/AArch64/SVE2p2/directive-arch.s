// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch armv9-a+sve2p2
bfcvtnt z23.h, p3/z, z13.s
// CHECK: bfcvtnt z23.h, p3/z, z13.s