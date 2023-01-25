// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sme2p1
sqcvt z0.h, { z0.s, z1.s }
// CHECK: sqcvt z0.h, { z0.s, z1.s }
