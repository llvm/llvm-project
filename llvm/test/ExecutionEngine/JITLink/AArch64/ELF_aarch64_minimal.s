# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=aarch64-unknown-linux-gnu -relax-relocations=false -position-independent -filetype=obj -o %t/aarch64_reloc.o %s
# RUN: llvm-jitlink -noexec %t/aarch64_reloc.o

	.text
	.globl	sub1
	.p2align	2
	.type	sub1,@function
sub1:
	sub	sp, sp, #16
	str	w0, [sp, #12]
	ldr	w8, [sp, #12]
	subs	w0, w8, #1
	add	sp, sp, #16
	ret

	.size	sub1, .-sub1

	.globl	main
	.p2align	2
	.type	main,@function
main:
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]
	add	x29, sp, #16
	stur	wzr, [x29, #-4]
	str	w0, [sp, #8]
	str	x1, [sp]
	ldr	w0, [sp, #8]
	bl	sub1
	ldp	x29, x30, [sp, #16]
	add	sp, sp, #32
	ret

	.size	main, .-main
