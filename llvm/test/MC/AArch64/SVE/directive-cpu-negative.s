// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.cpu generic+sve+nosve
ptrue   p0.b, pow2
// CHECK: error: instruction requires: sve or sme
// CHECK-NEXT: ptrue   p0.b, pow2
