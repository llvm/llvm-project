// RUN: llvm-mc -triple x86_64_lfi %s | FileCheck %s

movq %fs:0, %rax
// CHECK: movq 16(%r15), %rax

movq %fs:0, %rdi
// CHECK: movq 16(%r15), %rdi

movq %fs:0, %rcx
// CHECK: movq 16(%r15), %rcx
