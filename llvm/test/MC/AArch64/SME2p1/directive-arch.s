// RUN: llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

// SME2p1 should imply SME2
.arch armv9-a+sme2p1
sqcvt z0.h, {z0.s, z1.s}
// CHECK: sqcvt z0.h, { z0.s, z1.s }

.arch armv9-a+nosme2p1
