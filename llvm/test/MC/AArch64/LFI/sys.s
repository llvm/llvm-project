// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

svc #0
// CHECK:      mov x26, x30
// CHECK-NEXT: ldur x30, [x27, #-8]
// CHECK-NEXT: blr x30
// CHECK-NEXT: add x30, x27, w26, uxtw
