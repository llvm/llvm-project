// The purpose of this test is to verify that bss sections are emitted correctly.

// RUN: llvm-mc -filetype=obj -triple i686-apple-darwin9 %s | llvm-readobj -S | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple x86_64-apple-darwin9 %s | llvm-readobj -S | FileCheck %s

    .bss
    .globl _g0
    .align 4
_g0:
    .long 0

// CHECK:		Name: __bss (5F 5F 62 73 73 00 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:	Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:	Address: 0x0
// CHECK-NEXT:	Size: 0x4
// CHECK-NEXT:	Offset: 0
// CHECK-NEXT:	Alignment: 4
