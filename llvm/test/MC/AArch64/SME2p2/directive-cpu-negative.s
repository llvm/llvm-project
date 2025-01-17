// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.cpu generic+sme2p2
.cpu generic+nosme2p2
fmul    { z28.h - z31.h }, { z28.h - z31.h }, z15.h
// CHECK: error: instruction requires: sme2p2
// CHECK: fmul    { z28.h - z31.h }, { z28.h - z31.h }, z15.h

.cpu generic+sme-tmop
.cpu generic+nosme-tmop
sutmopa za1.s, {z10.b-z11.b}, z21.b, z29[1]
// CHECK: error: instruction requires: sme-tmop
// CHECK: sutmopa za1.s, {z10.b-z11.b}, z21.b, z29[1]

.cpu generic+sme-mop4
.cpu generic+nosme-mop4
usmop4s za0.s, z0.b, z16.b
// CHECK: error: instruction requires: sme-mop4
// CHECK: usmop4s za0.s, z0.b, z16.b
