// The purpose of this test is to verify that bss sections are emitted correctly.

// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | llvm-readobj -S - | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | llvm-readobj -S - | FileCheck %s

    .bss
    .globl _g0
    .align 4
_g0:
    .long 0

// CHECK:      Name:            .bss
// CHECK-NEXT: VirtualSize:     0
// CHECK-NEXT: VirtualAddress:  0
// CHECK-NEXT: RawDataSize:     4
