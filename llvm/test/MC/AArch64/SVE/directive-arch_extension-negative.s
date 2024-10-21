// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sve2+nosve

ptrue   p0.b, pow2
// CHECK: error: instruction requires: sve or sme
// CHECK-NEXT: ptrue   p0.b, pow2

// Check that setting +nosve implies +nosve2
adclb z0.s, z1.s, z31.s
// CHECK: error: instruction requires: sve2
// CHECK-NEXT: adclb z0.s, z1.s, z31.s
