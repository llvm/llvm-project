// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sve2
.arch_extension nosve2
tbx z0.b, z1.b, z2.b
// CHECK: error: instruction requires: sve2 or sme
// CHECK-NEXT: tbx z0.b, z1.b, z2.b

.arch_extension sve-aes
.arch_extension nosve-aes
aesd z23.b, z23.b, z13.b
// CHECK: error: instruction requires: sve2 or ssve-aes sve-aes
// CHECK-NEXT: aesd z23.b, z23.b, z13.b

// nosve2-aes should disable sve-aes but not sve2.
.arch_extension sve2-aes
.arch_extension nosve2-aes
aesd z23.b, z23.b, z13.b
// CHECK: error: instruction requires: sve-aes
// CHECK-NEXT: aesd z23.b, z23.b, z13.b

.arch_extension sve2-sm4
.arch_extension nosve2-sm4
sm4e z0.s, z0.s, z0.s
// CHECK: error: instruction requires: sve2-sm4
// CHECK-NEXT: sm4e z0.s, z0.s, z0.s

.arch_extension sve2-sha3
.arch_extension nosve2-sha3
rax1 z0.d, z0.d, z0.d
// CHECK: error: instruction requires: sve2-sha3
// CHECK-NEXT: rax1 z0.d, z0.d, z0.d

.arch_extension sve2-bitperm
.arch_extension nosve2-bitperm
bgrp z21.s, z10.s, z21.s
// CHECK: error: instruction requires: sve2-bitperm
// CHECK-NEXT: bgrp z21.s, z10.s, z21.s

.arch_extension f8f16mm
.arch_extension nof8f16mm
fmmla   z23.h, z13.b, z8.b
// CHECK: error: instruction requires: f8f16mm
// CHECK-NEXT: fmmla   z23.h, z13.b, z8.b

.arch_extension f8f32mm
.arch_extension nof8f32mm
fmmla   z23.s, z13.b, z8.b
// CHECK: error: instruction requires: f8f32mm
// CHECK-NEXT: fmmla   z23.s, z13.b, z8.b

.arch_extension sve-f16f32mm
.arch_extension nosve-f16f32mm
fmmla   z23.s, z13.h, z8.h
// CHECK: error: instruction requires: sve-f16f32mm
// CHECK-NEXT: fmmla   z23.s, z13.h, z8.h

.arch_extension sve-bfscale
.arch_extension nosve-bfscale
bfscale z0.h, p0/m, z0.h, z0.h
// CHECK: error: instruction requires: sve-bfscale
// CHECK-NEXT: bfscale z0.h, p0/m, z0.h, z0.h
