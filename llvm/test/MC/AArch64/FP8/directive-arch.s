// RUN: llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

.arch armv9-a+fp8
bf1cvtl v0.8h, v0.8b
// CHECK: bf1cvtl v0.8h, v0.8b
.arch armv9-a+nofp8

.arch armv9-a+faminmax
famax  v31.4h, v31.4h, v31.4h
// CHECK: famax  v31.4h, v31.4h, v31.4h
.arch armv9-a+nofaminmax

.arch armv9-a+fp8fma
fmlalb  v0.8h, v0.16b, v0.16b
// CHECK: fmlalb  v0.8h, v0.16b, v0.16b
.arch armv9-a+nofp8fma

.arch armv9-a+ssve-fp8fma
fmlalb z23.h, z13.b, z0.b[7]
// CHECK: fmlalb z23.h, z13.b, z0.b[7]
.arch armv9-a+nossve-fp8fma

.arch armv9-a+fp8dot2
fdot  v31.4h, v0.8b, v0.8b
// CHECK: fdot  v31.4h, v0.8b, v0.8b
.arch armv9-a+nofp8dot2

.arch armv9-a+fp8dot4
fdot  v0.2s, v0.8b, v31.8b
// CHECK: fdot  v0.2s, v0.8b, v31.8b
.arch armv9-a+nofp8dot4

.arch armv9-a+lut
luti2  v30.8h, {v20.8h}, v31[7]
// CHECK: luti2  v30.8h, { v20.8h }, v31[7]
.arch armv9-a+nolut

.arch armv9-a+sve2+lut
luti2  z0.h, {z0.h}, z0[0]
// CHECK: luti2  z0.h, { z0.h }, z0[0]
.arch armv9-a+nosve2+nolut

.arch armv9-a+sme2p1+sme-lutv2
luti4  {z0.b-z3.b}, zt0, {z0-z1}
// CHECK: luti4  { z0.b - z3.b }, zt0, { z0, z1 }
.arch armv9-a+nosme2p1+nosme-lutv2
