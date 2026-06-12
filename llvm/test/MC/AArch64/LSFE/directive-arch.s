// RUN: llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

.arch armv9.6-a+lsfe
ldfadd h0, h1, [x2]
// CHECK: ldfadd h0, h1, [x2]