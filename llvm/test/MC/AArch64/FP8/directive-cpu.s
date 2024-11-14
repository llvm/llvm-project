// RUN: llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

.cpu generic+fp8
bf1cvtl v0.8h, v0.8b
// CHECK: bf1cvtl v0.8h, v0.8b

.cpu generic+faminmax
famax  v31.4h, v31.4h, v31.4h
// CHECK: famax  v31.4h, v31.4h, v31.4h

.cpu generic+fp8fma
fmlalb  v0.8h, v0.16b, v0.16b
// CHECK: fmlalb  v0.8h, v0.16b, v0.16b

.cpu generic+ssve-fp8fma
fmlalb z23.h, z13.b, z0.b[7]
// CHECK: fmlalb z23.h, z13.b, z0.b[7]

.cpu generic+fp8dot2
fdot  v31.4h, v0.8b, v0.8b
// CHECK: fdot  v31.4h, v0.8b, v0.8b

.cpu generic+fp8dot4
fdot  v0.2s, v0.8b, v31.8b
// CHECK: fdot  v0.2s, v0.8b, v31.8b

.cpu generic+sme-f16f16
fsub za.h[w10, 5, vgx2], {z10.h, z11.h}
// CHECK: fsub za.h[w10, 5, vgx2], { z10.h, z11.h }

.cpu generic+sme-f8f32
fdot za.s[w8, 0, vgx2], {z0.b-z1.b}, z0.b
// CHECK: fdot za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b

.cpu generic+sme-f8f16
fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.b
// CHECK: fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.b

.cpu generic+lut
luti2  v30.8h, {v20.8h}, v31[7]
// CHECK: luti2  v30.8h, { v20.8h }, v31[7]

.cpu generic+sve2+lut
luti2  z0.h, {z0.h}, z0[0]
// CHECK: luti2  z0.h, { z0.h }, z0[0]

.cpu generic+sme2p1+sme-lutv2
luti4  {z0.b-z3.b}, zt0, {z0-z1}
// CHECK: luti4  { z0.b - z3.b }, zt0, { z0, z1 }