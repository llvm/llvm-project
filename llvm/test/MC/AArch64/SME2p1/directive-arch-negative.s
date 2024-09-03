// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch armv9-a+sme2p1
.arch armv9-a+nosme2p1
sqcvt z0.h, {z0.s, z1.s}
// CHECK: error: instruction requires: sme2
// CHECK: sqcvt z0.h, {z0.s, z1.s}

.arch armv9-a+sme2+sve-b16b16
.arch armv9-a+sme2+nosve-b16b16
bfclamp { z0.h, z1.h }, z0.h, z0.h
// CHECK: error: instruction requires: sve-b16b16
// CHECK: bfclamp { z0.h, z1.h }, z0.h, z0.h

.arch armv9-a+sme-b16b16
.arch armv9-a+nosme-b16b16
bfadd za.h[w8, 3], {z20.h-z21.h}
// CHECK: error: instruction requires: sme-b16b16
// CHECK: bfadd za.h[w8, 3], {z20.h-z21.h} 
