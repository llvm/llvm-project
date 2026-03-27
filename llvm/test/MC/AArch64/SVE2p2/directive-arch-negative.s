// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch armv9-a+sve2p2
.arch armv9-a+nosve2p2
bfcvtnt z23.h, p3/z, z13.s
// CHECK: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: bfcvtnt z23.h, p3/z, z13.s