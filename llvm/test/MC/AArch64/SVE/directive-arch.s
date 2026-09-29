// RUN: llvm-mc -triple=aarch64 < %s | FileCheck %s

.arch armv8-a+sve

ptrue   p0.b, pow2
// CHECK: ptrue   p0.b, pow2

// Test that the implied +sve feature is also set from +sve2.
.arch armv8-a+sve2

ptrue   p0.b, pow2
// CHECK: ptrue   p0.b, pow2
