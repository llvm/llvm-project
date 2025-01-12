// RUN: not llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

// SME2 should require SME
.arch armv9-a+sme2+nosme
sqcvt z0.h, {z0.s, z1.s}
// CHECK: error: instruction requires: sme2
// CHECK-NEXT: sqcvt z0.h, {z0.s, z1.s}

.arch armv9-a+sme2+nosme2
sqcvt z0.h, {z0.s, z1.s}
// CHECK: error: instruction requires: sme2
// CHECK-NEXT: sqcvt z0.h, {z0.s, z1.s}

.arch armv9-a+sme2+sve-b16b16+nosve-b16b16
bfclamp {z0.h, z1.h}, z0.h, z0.h
// CHECK: error: instruction requires: sve-b16b16
// CHECK-NEXT: bfclamp {z0.h, z1.h}, z0.h, z0.h

.arch armv9-a+sme-b16b16+nosme-b16b16
bfadd za.h[w8, 3], {z20.h-z21.h}
// CHECK: error: instruction requires: sme-b16b16
// CHECK-NEXT: bfadd za.h[w8, 3], {z20.h-z21.h}