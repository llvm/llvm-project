// RUN: llvm-mc -triple=aarch64 < %s | FileCheck %s

.arch_extension sve

ptrue   p0.b, pow2
// CHECK: ptrue   p0.b, pow2
