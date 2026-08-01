// RUN: llvm-mc -triple x86_64_lfi %s | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple x86_64_lfi %s | llvm-readelf -S - | FileCheck %s --check-prefix=ELF

// CHECK:      .section .note.LFI.ABI.x86_64,"aG",@note,.note.LFI.ABI.x86_64,comdat
// CHECK-NEXT: .long 4
// CHECK-NEXT: .long 7
// CHECK-NEXT: .long 1
// CHECK-NEXT: .ascii "LFI"
// CHECK-NEXT: .byte 0
// CHECK-NEXT: .p2align 2, 0x0
// CHECK-NEXT: .ascii "x86_64"
// CHECK-NEXT: .byte 0
// CHECK-NEXT: .p2align 2, 0x0

// ELF: .note.LFI.ABI.x86_64 NOTE {{.*}} AG
