// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s -o - | llvm-readobj -s | FileCheck %s

	.def	 _main;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	_main
_main:
	.cfi_startproc
	ret
	.cfi_endproc

// CHECK:    Name: .eh_frame
