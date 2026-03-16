// RUN: llvm-mc -triple x86_64_lfi %s | FileCheck %s

syscall
// CHECK:      leaq .Ltmp0(%rip), %r11
// CHECK-NEXT: jmpq *(%r14)
// CHECK-NEXT: .Ltmp0:

movq %fs:0, %rax
// CHECK: movq 16(%r15), %rax

movq %fs:0, %rdi
// CHECK: movq 16(%r15), %rdi
