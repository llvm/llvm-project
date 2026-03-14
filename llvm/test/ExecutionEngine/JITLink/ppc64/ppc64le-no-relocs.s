# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-jitlink -noexec %t
#
# Check that a program that just returns immediately from main (requiring no
# relocations at all) loads under llvm-jitlink.

	.text
	.abiversion 2
	.file	"ppc64le-no-relocs.c"
	.globl	main
	.p2align	4
	.type	main,@function
main:
.Lfunc_begin0:
	li 3, 0
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	main, .Lfunc_end0-.Lfunc_begin0
