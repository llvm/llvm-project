// RUN: llvm-mc -triple=aarch64 < %s | FileCheck %s

.arch_extension sve

ptrue   p0.b, pow2
// CHECK: ptrue   p0.b, pow2

// Test that the implied +sve feature is also set from +sve2.
.arch_extension nosve
.arch_extension sve2
ptrue   p0.b, pow2
// CHECK: ptrue   p0.b, pow2

// Check that setting +nosve2 does not imply +nosve
.arch_extension nosve2

ptrue   p0.b, pow2
// CHECK: ptrue   p0.b, pow2
