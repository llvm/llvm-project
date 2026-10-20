// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.cpu generic+sve2p2
fcvtnt  z0.s, p0/z, z0.d
// CHECK: fcvtnt  z0.s, p0/z, z0.d