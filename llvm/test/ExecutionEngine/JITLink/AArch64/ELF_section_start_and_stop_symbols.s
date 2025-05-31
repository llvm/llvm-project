# RUN: llvm-mc -triple=aarch64-unknown-linux-gnu -position-independent \
# RUN:   -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -check %s %t.o

	.text
	.file	"elf_section_start_stop.c"
	.globl	main
	.p2align	2
	.type	main,@function
main:
	adrp	x8, z
	adrp	x9, y
	ldr	w8, [x8, :lo12:z]
	ldr	w9, [x9, :lo12:y]
	sub	w0, w8, w9
	ret
.Lfunc_end0:
	.size	main, .Lfunc_end0-main

	.type	x,@object
	.section	custom_section,"aw",@progbits
	.globl	x
	.p2align	2
x:
	.word	42
	.size	x, 4

# jitlink-check: *{8}z = (*{8}y) + 4

	.type	y,@object
	.data
	.globl	y
	.p2align	3, 0x0
y:
	.xword	__start_custom_section
	.size	y, 8

	.type	z,@object
	.globl	z
	.p2align	3, 0x0
z:
	.xword	__stop_custom_section
	.size	z, 8
