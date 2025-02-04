// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch armv9-a+sve2
.arch armv9-a+nosve2
tbx z0.b, z1.b, z2.b
// CHECK: error: instruction requires: sve2 or sme
// CHECK-NEXT: tbx z0.b, z1.b, z2.b

.arch armv9-a+sve-aes+nosve-aes
aesd z23.b, z23.b, z13.b
// CHECK: error: instruction requires: sve-aes
// CHECK-NEXT: aesd z23.b, z23.b, z13.b

// nosve2-aes should disable sve-aes but not sve2.
.arch armv9-a+sve2-aes+nosve2-aes
aesd z23.b, z23.b, z13.b
// CHECK: error: instruction requires: sve-aes
// CHECK-NEXT: aesd z23.b, z23.b, z13.b

.arch armv9-a+sve2-sm4
.arch armv9-a+nosve2-sm4
sm4e z0.s, z0.s, z0.s
// CHECK: error: instruction requires: sve2-sm4
// CHECK-NEXT: sm4e z0.s, z0.s, z0.s

.arch armv9-a+sve2-sha3
.arch armv9-a+nosve2-sha3
rax1 z0.d, z0.d, z0.d
// CHECK: error: instruction requires: sve2-sha3
// CHECK-NEXT: rax1 z0.d, z0.d, z0.d

.arch armv9-a+ssve-bitperm
.arch armv9-a+nossve-bitperm
bgrp z21.s, z10.s, z21.s
// CHECK: error: instruction requires: sve-bitperm
// CHECK-NEXT: bgrp z21.s, z10.s, z21.s

.arch armv9-a+sve2+sve-bitperm
.arch armv9-a+sve2+nosve-bitperm
bgrp z21.s, z10.s, z21.s
// CHECK: error: instruction requires: sve-bitperm
// CHECK-NEXT: bgrp z21.s, z10.s, z21.s

.arch armv9-a+f8f16mm
.arch armv9-a+nof8f16mm
fmmla   z23.h, z13.b, z8.b
// CHECK: error: instruction requires: f8f16mm
// CHECK-NEXT: fmmla   z23.h, z13.b, z8.b

.arch armv9-a+f8f32mm
.arch armv9-a+nof8f32mm
fmmla   z23.s, z13.b, z8.b
// CHECK: error: instruction requires: f8f32mm
// CHECK-NEXT: fmmla   z23.s, z13.b, z8.b

.arch armv9-a+sve-f16f32mm
.arch armv9-a+nosve-f16f32mm
fmmla   z23.s, z13.h, z8.h
// CHECK: error: instruction requires: sve-f16f32mm
// CHECK-NEXT: fmmla   z23.s, z13.h, z8.h

.arch armv9-a+sve-bfscale
.arch armv9-a+nosve-bfscale
bfscale z0.h, p0/m, z0.h, z0.h
// CHECK: error: instruction requires: sve-bfscale
// CHECK-NEXT: bfscale z0.h, p0/m, z0.h, z0.h
