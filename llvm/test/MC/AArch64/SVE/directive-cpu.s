// RUN: llvm-mc -triple=aarch64 < %s | FileCheck %s

.cpu generic+sve
ptrue   p0.b, pow2
// CHECK: ptrue   p0.b, pow2

// Test that the implied +sve feature is also set from +sve2.
.cpu generic+sve2
ptrue   p0.b, pow2
// CHECK: ptrue   p0.b, pow2

// Check that setting +nosve2 does not imply +nosve
.cpu generic+sve2+nosve2
ptrue   p0.b, pow2
// CHECK: ptrue   p0.b, pow2
