// RUN: llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

.arch armv9-a+sme2p2
fmul    { z28.h - z31.h }, { z28.h - z31.h }, z15.h
// CHECK: fmul    { z28.h - z31.h }, { z28.h - z31.h }, z15.h

.arch armv9-a+sme-tmop
ftmopa  za0.s, {z0.s-z1.s}, z0.s, z20[0]
// CHECK: ftmopa za0.s, { z0.s, z1.s }, z0.s, z20[0]

.arch armv9-a+sme-mop4
bfmop4a za0.s, z0.h, z16.h
// CHECK: bfmop4a za0.s, z0.h, z16.h
