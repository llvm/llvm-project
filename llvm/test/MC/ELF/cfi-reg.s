// RUN: llvm-mc -triple x86_64-pc-linux-gnu %s -o - | FileCheck %s
// PR13754

f:
	.cfi_startproc
        nop
	.cfi_offset 6, -16
        nop
	.cfi_offset %rsi, -16
        nop
	.cfi_offset rbx, -16
        nop
	.cfi_endproc

// CHECK: f:
// CHECK: .cfi_offset %rbp, -16
// CHECK: .cfi_offset %rsi, -16
// CHECK: .cfi_offset %rbx, -16
