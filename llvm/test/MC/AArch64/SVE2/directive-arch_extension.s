// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sve2
tbx z0.b, z1.b, z2.b
// CHECK: tbx z0.b, z1.b, z2.b

.arch_extension sve-aes
aesd z23.b, z23.b, z13.b
// CHECK: aesd z23.b, z23.b, z13.b

.arch_extension sve2-aes
aesd z23.b, z23.b, z13.b
// CHECK: aesd z23.b, z23.b, z13.b

.arch_extension sve2-sm4
sm4e z0.s, z0.s, z0.s
// CHECK: sm4e z0.s, z0.s, z0.s

.arch_extension sve2-sha3
rax1 z0.d, z0.d, z0.d
// CHECK: rax1 z0.d, z0.d, z0.d

.arch_extension sve2-bitperm
bgrp z21.s, z10.s, z21.s
// CHECK: bgrp z21.s, z10.s, z21.s

.arch_extension f8f16mm
fmmla   z23.h, z13.b, z8.b
// CHECK: fmmla   z23.h, z13.b, z8.b

.arch_extension f8f32mm
fmmla   z23.s, z13.b, z8.b
// CHECK: fmmla   z23.s, z13.b, z8.b

.arch_extension sve-f16f32mm
fmmla   z23.s, z13.h, z8.h
// CHECK: fmmla   z23.s, z13.h, z8.h

.arch_extension sve-bfscale
bfscale z0.h, p0/m, z0.h, z0.h
// CHECK: bfscale z0.h, p0/m, z0.h, z0.h
