// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu %s -o - | llvm-readobj --symbols - | FileCheck %s

	.text
	.globl	foobar
	.align	16, 0x90
	.type	foobar,@function
foobar:
	pushl	%ebp
	movl	%esp, %ebp
	subl	$8, %esp
	calll	foo
	calll	bar
	addl	$8, %esp
	popl	%ebp
	retl
.Ltmp3:
	.size	foobar, .Ltmp3-foobar

// CHECK:     Name: foobar (11)
// CHECK:     Name: foo (18)
// CHECK:     Name: bar (14)
