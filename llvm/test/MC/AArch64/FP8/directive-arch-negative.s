// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch armv9-a+fp8
.arch armv9-a+nofp8
bf1cvtl v0.8h, v0.8b
// CHECK: error: instruction requires: fp8
// CHECK: bf1cvtl v0.8h, v0.8b

.arch armv9-a+faminmax
.arch armv9-a+nofaminmax
famax  v31.4h, v31.4h, v31.4h
// CHECK: error: instruction requires: faminmax
// CHECK: famax  v31.4h, v31.4h, v31.4h

.arch armv9-a+fp8fma
.arch armv9-a+nofp8fma
fmlalb  v0.8h, v0.16b, v0.16b
// CHECK: error: instruction requires: fp8fma
// CHECK: fmlalb  v0.8h, v0.16b, v0.16b

.arch armv9-a+ssve-fp8fma
.arch armv9-a+nossve-fp8fma
fmlalb  z23.h, z13.b, z0.b[7]
// CHECK: error: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK: fmlalb  z23.h, z13.b, z0.b[7]

.arch armv9-a+fp8dot2
.arch armv9-a+nofp8dot2
fdot  v31.4h, v0.8b, v0.8b
// CHECK: error: instruction requires: fp8dot2
// CHECK: fdot  v31.4h, v0.8b, v0.8b

.arch armv9-a+fp8dot4
.arch armv9-a+nofp8dot4
fdot  v0.2s, v0.8b, v31.8b
// CHECK: error: instruction requires: fp8dot4
// CHECK: fdot  v0.2s, v0.8b, v31.8b
