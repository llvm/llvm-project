// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sme2p1
.arch_extension nosme2
sqcvt z0.h, { z0.s, z1.s }
// CHECK: error: instruction requires: sme2
// CHECK: sqcvt z0.h
