// RUN: not llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

.arch_extension fp8
.arch_extension nofp8
bf1cvtl v0.8h, v0.8b
// CHECK: error: instruction requires: fp8
// CHECK-NEXT: bf1cvtl v0.8h, v0.8b

.arch_extension faminmax
.arch_extension nofaminmax
famax  v31.4h, v31.4h, v31.4h
// CHECK: error: instruction requires: faminmax
// CHECK-NEXT: famax  v31.4h, v31.4h, v31.4h

.arch_extension fp8fma
.arch_extension nofp8fma
fmlalb  v0.8h, v0.16b, v0.16b
// CHECK: error: instruction requires: fp8fma
// CHECK-NEXT: fmlalb  v0.8h, v0.16b, v0.16b

.arch_extension ssve-fp8fma
.arch_extension nossve-fp8fma
fmlalb z23.h, z13.b, z0.b[7]
// CHECK: error: instruction requires: ssve-fp8fma
// CHECK-NEXT: fmlalb z23.h, z13.b, z0.b[7]

.arch_extension fp8dot2
.arch_extension nofp8dot2
fdot  v31.4h, v0.8b, v0.8b
// CHECK: error: instruction requires: fp8dot2
// CHECK-NEXT: fdot  v31.4h, v0.8b, v0.8b

.arch_extension fp8dot4
.arch_extension nofp8dot4
fdot  v0.2s, v0.8b, v31.8b
// CHECK: error: instruction requires: fp8dot4
// CHECK-NEXT: fdot  v0.2s, v0.8b, v31.8b

.arch_extension sme-f16f16
.arch_extension nosme-f16f16
fsub za.h[w10, 5, vgx2], {z10.h, z11.h}
// CHECK: error: instruction requires: sme-f16f16 or sme-f8f16
// CHECK-NEXT: fsub za.h[w10, 5, vgx2], {z10.h, z11.h}

.arch_extension sme-f8f32
.arch_extension nosme-f8f32
fdot za.s[w8, 0, vgx2], {z0.b-z1.b}, z0.b
// CHECK: error: instruction requires: sme-f8f32
// CHECK-NEXT: fdot za.s[w8, 0, vgx2], {z0.b-z1.b}, z0.b

.arch_extension sme-f8f16
.arch_extension nosme-f8f16
fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.b
// CHECK: error: instruction requires: sme-f8f16
// CHECK-NEXT: fdot za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.b

.arch_extension lut
.arch_extension nolut
luti2  v30.8h, {v20.8h}, v31[7]
// CHECK: error: instruction requires: lut
// CHECK-NEXT: luti2  v30.8h, {v20.8h}, v31[7]

.arch_extension sme2p1
.arch_extension sme-lutv2
.arch_extension nosme-lutv2
luti4  {z0.b-z3.b}, zt0, {z0-z1}
// CHECK: error: instruction requires: sme-lutv2
// CHECK-NEXT: luti4 {z0.b-z3.b}, zt0, {z0-z1}