// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sme2p1
sqcvt z0.h, { z0.s, z1.s }
// CHECK: sqcvt z0.h, { z0.s, z1.s }

.arch_extension sme2
.arch_extension sve-b16b16
bfclamp { z0.h, z1.h }, z0.h, z0.h
// CHECK: bfclamp { z0.h, z1.h }, z0.h, z0.h