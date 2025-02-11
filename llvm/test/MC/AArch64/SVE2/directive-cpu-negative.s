// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.cpu generic+sve2
.cpu generic+nosve2
tbx z0.b, z1.b, z2.b
// CHECK: error: instruction requires: sve2 or sme
// CHECK-NEXT: tbx z0.b, z1.b, z2.b

// nosve2-aes should disable sve-aes but not sve2.
.cpu generic+sve2-aes+nosve2-aes
aesd z23.b, z23.b, z13.b
// CHECK: error: instruction requires: sve-aes
// CHECK-NEXT: aesd z23.b, z23.b, z13.b

.cpu generic+sve-aes+nosve-aes
aesd z23.b, z23.b, z13.b
// CHECK: error: instruction requires: sve2 or ssve-aes sve-aes
// CHECK-NEXT: aesd z23.b, z23.b, z13.b

.cpu generic+sve2-sm4
.cpu generic+nosve2-sm4
sm4e z0.s, z0.s, z0.s
// CHECK: error: instruction requires: sve2-sm4
// CHECK-NEXT: sm4e z0.s, z0.s, z0.s

.cpu generic+sve2-sha3
.cpu generic+nosve2-sha3
rax1 z0.d, z0.d, z0.d
// CHECK: error: instruction requires: sve2-sha3
// CHECK-NEXT: rax1 z0.d, z0.d, z0.d

.cpu generic+sve2+sve-bitperm
.cpu generic+sve2+nosve-bitperm
bgrp z21.s, z10.s, z21.s
// CHECK: error: instruction requires: sve-bitperm
// CHECK-NEXT: bgrp z21.s, z10.s, z21.s

.cpu generic+ssve-bitperm
.cpu generic+nossve-bitperm
bgrp z21.s, z10.s, z21.s
// CHECK: error: instruction requires: sve2 or ssve-bitperm sve-bitperm
// CHECK-NEXT: bgrp z21.s, z10.s, z21.s

.cpu generic+sve2+f8f16mm
.cpu generic+sve2+nof8f16mm
fmmla   z23.h, z13.b, z8.b
// CHECK: error: instruction requires: f8f16mm
// CHECK-NEXT: fmmla   z23.h, z13.b, z8.b

.cpu generic+sve2+f8f32mm
.cpu generic+sve2+nof8f32mm
fmmla   z23.s, z13.b, z8.b
// CHECK: error: instruction requires: f8f32mm
// CHECK-NEXT: fmmla   z23.s, z13.b, z8.b

.cpu generic+sve-f16f32mm
.cpu generic+nosve-f16f32mm
fmmla   z23.s, z13.h, z8.h
// CHECK: error: instruction requires: sve-f16f32mm
// CHECK-NEXT: fmmla   z23.s, z13.h, z8.h

.cpu generic+sve-bfscale
.cpu generic+nosve-bfscale
bfscale z0.h, p0/m, z0.h, z0.h
// CHECK: error: instruction requires: sve-bfscale
// CHECK-NEXT: bfscale z0.h, p0/m, z0.h, z0.h
