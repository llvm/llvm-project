// RUN: llvm-mc -triple x86_64_lfi %s | FileCheck %s

syscall
// CHECK:      .bundle_lock
// CHECK-NEXT: leaq .Ltmp0(%rip), %r11
// CHECK-NEXT: jmpq *-8(%r14)
// CHECK-NEXT: .Ltmp0:
// CHECK-NEXT: .bundle_unlock
