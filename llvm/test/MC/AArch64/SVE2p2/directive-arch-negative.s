// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

// SVE2p2 should require SVE2p1
.arch armv9-a+sve2p2+nosve2p1
bfcvtnt z23.h, p3/z, z13.s
// CHECK: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: bfcvtnt z23.h, p3/z, z13.s

.arch armv9-a+sve2p2+nosve2p2
bfcvtnt z23.h, p3/z, z13.s
// CHECK: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: bfcvtnt z23.h, p3/z, z13.s