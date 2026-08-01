// RUN: llvm-mc -triple x86_64_lfi %s | FileCheck %s

syscall
// CHECK:      leaq .Ltmp0(%rip), %r11
// CHECK-NEXT: jmpq *-8(%r14)
// CHECK-NEXT: .Ltmp0:
