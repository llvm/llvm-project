// REQUIRES: x86-registered-target
// RUN: llvm-mc -triple i686-unknown-linux-gnu -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-ASM-ROUNDTRIP
// RUN: llvm-mc -triple i686-unknown-linux-gnu -filetype obj -o - %s | llvm-objdump -dwarf=frames - | FileCheck %s -check-prefix CHECK-EH_FRAME

	.text

	.section .text.f,"ax",@progbits
	.global f
	.type f,@function
f:
	.cfi_startproc
	.cfi_return_column 0
	.cfi_endproc

	.section .text.g,"ax",@progbits
	.global g
	.type g,@function
g:
	.cfi_startproc
	.cfi_return_column 65
	.cfi_endproc

	.section .text.h,"ax",@progbits
	.global h
	.type g,@function
h:
	.cfi_startproc
	.cfi_return_column 65
	.cfi_endproc

// CHECK-ASM-ROUNDTRIP-LABEL: f:
// CHECK-ASM-ROUNDTRIP: .cfi_startproc
// CHECK-ASM-ROUNDTRIP-NEXT: .cfi_return_column 0
// CHECK-ASM-ROUNDTRIP: .cfi_endproc

// CHECK-EH_FRAME: 00000000 00000014 ffffffff CIE
// CHECK-EH_FRAME:   Return address column: 0

// CHECK-EH_FRAME: 0000002c 00000014 ffffffff CIE
// CHECK-EH_FRAME:   Return address column: 65

// CHECK-EH_FRAME-NOT: ........ 00000014 ffffffff CIE

