// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

// CHECK:      .section .note.LFI.ABI.aarch64,"aG",@note,.note.LFI.ABI.aarch64,comdat
// CHECK-NEXT: .word 4
// CHECK-NEXT: .word 8
// CHECK-NEXT: .word 1
// CHECK-NEXT: .ascii "LFI"
// CHECK-NEXT: .byte 0
// CHECK-NEXT: .p2align 2, 0x0
// CHECK-NEXT: .ascii "aarch64"
// CHECK-NEXT: .byte 0
// CHECK-NEXT: .p2align 2, 0x0
// CHECK-NEXT: .text
