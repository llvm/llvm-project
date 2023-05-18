// RUN: llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s


.arch armv9-a+sme2
add za.s[w8, 7], {z20.s-z21.s}, z10.s
// CHECK: add	za.s[w8, 7, vgx2], { z20.s, z21.s }, z10.s

.arch armv9-a+nosme2

