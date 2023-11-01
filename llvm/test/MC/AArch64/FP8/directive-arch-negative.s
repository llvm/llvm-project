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
