// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.cpu generic+fp8+nofp8
bf1cvtl v0.8h, v0.8b
// CHECK: error: instruction requires: fp8
// CHECK: bf1cvtl v0.8h, v0.8b

.cpu generic+faminmax+nofaminmax
famax  v31.4h, v31.4h, v31.4h
// CHECK: error: instruction requires: faminmax
// CHECK: famax  v31.4h, v31.4h, v31.4h

.cpu generic+fp8fma+nofp8fma
fmlalb  v0.8h, v0.16b, v0.16b
// CHECK: error: instruction requires: fp8fma
// CHECK: fmlalb  v0.8h, v0.16b, v0.16b

.cpu generic+ssve-fp8fma+nossve-fp8fma
fmlalb  z23.h, z13.b, z0.b[7]
// CHECK: error: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK: fmlalb  z23.h, z13.b, z0.b[7]

.cpu generic+fp8dot2+nofp8dot2
fdot  v31.4h, v0.8b, v0.8b
// CHECK: error: instruction requires: fp8dot2
// CHECK: fdot  v31.4h, v0.8b, v0.8b

.cpu generic+fp8dot4+nofp8dot4
fdot  v0.2s, v0.8b, v31.8b
// CHECK: error: instruction requires: fp8dot4
// CHECK: fdot  v0.2s, v0.8b, v31.8b

.cpu generic+sme-f16f16+nosme-f16f16
fsub za.h[w10, 5, vgx2], {z10.h, z11.h}
// CHECK: error: instruction requires: sme-f16f16 or sme-f8f16
// CHECK-NEXT: fsub za.h[w10, 5, vgx2], {z10.h, z11.h}

.cpu generic+sme-f8f32+nosme-f8f32
fdot za.s[w8, 0, vgx2], {z0.b-z1.b}, z0.b
// CHECK: error: instruction requires: sme-f8f32
// CHECK-NEXT: fdot za.s[w8, 0, vgx2], {z0.b-z1.b}, z0.b

.cpu generic+sme-f8f16+nosme-f8f16
fdot za.h[w8, 0, vgx2], {z0.b, z1.b}, z0.b
// CHECK: error: instruction requires: sme-f8f16
// CHECK-NEXT: fdot za.h[w8, 0, vgx2], {z0.b, z1.b}, z0.b

.cpu generic+lut+nolut
luti2  v30.8h, { v20.8h }, v31[7]
// CHECK: error: instruction requires: lut
// CHECK: luti2  v30.8h, { v20.8h }, v31[7]

.cpu generic+sve2+lut+nosve2+nolut
luti2  z0.h, { z0.h }, z0[0]
// CHECK: error: instruction requires: lut sve2 or sme2
// CHECK: luti2  z0.h, { z0.h }, z0[0]

.cpu generic+sme-lutv2+nosme-lutv2
luti4  { z0.b - z3.b }, zt0, { z0, z1 }
// CHECK: error: instruction requires: sme-lutv2
// CHECK: luti4  { z0.b - z3.b }, zt0, { z0, z1 }