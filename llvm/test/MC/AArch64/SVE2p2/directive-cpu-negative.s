// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.cpu generic+sve2p2
.cpu generic+nosve2p2
fcvtnt  z0.s, p0/z, z0.d
// CHECK: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: fcvtnt  z0.s, p0/z, z0.d