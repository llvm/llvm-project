// RUN: llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

.arch armv9-a+fp8
bf1cvtl v0.8h, v0.8b
// CHECK: bf1cvtl v0.8h, v0.8b

.arch armv9-a+nofp8
.arch armv9-a+faminmax
famax  v31.4h, v31.4h, v31.4h
// CHECK: famax  v31.4h, v31.4h, v31.4h

.arch armv9-a+nofaminmax

