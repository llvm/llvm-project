// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sme2p2
.arch_extension nosme2p2
ftmopa  za0.s, {z0.s-z1.s}, z0.s, z20[0]
// CHECK: error: instruction requires: sme2p2
// CHECK: ftmopa  za0.s, {z0.s-z1.s}, z0.s, z20[0]
