// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

// Check that setting +nosve implies +nosve2
.arch armv9-a+nosve

adclb z0.s, z1.s, z31.s
// CHECK: error: instruction requires: sve2
// CHECK-NEXT: adclb z0.s, z1.s, z31.s
