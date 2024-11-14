// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

// SVE2p2 should require SVE2p1
.cpu generic+sve2p2+nosve2p1
fcvtnt  z0.s, p0/z, z0.d
// CHECK: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: fcvtnt  z0.s, p0/z, z0.d

.cpu generic+sve2p2+nosve2p2
fcvtnt  z0.s, p0/z, z0.d
// CHECK: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: fcvtnt  z0.s, p0/z, z0.d