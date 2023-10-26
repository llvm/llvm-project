// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch armv9-a+fp8
.arch armv9-a+nofp8
bf1cvtl v0.8h, v0.8b
// CHECK: error: instruction requires: fp8
// CHECK: bf1cvtl v0.8h, v0.8b
