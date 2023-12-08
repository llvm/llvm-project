// RUN: llvm-mc -triple i686-unknown-unknown -x86-asm-syntax=intel %s | FileCheck %s

// CHECK: leaw	(%bp,%si), %ax
lea ax, [bp+si]
// CHECK: leaw	(%bp,%si), %ax
lea ax, [si+bp]
