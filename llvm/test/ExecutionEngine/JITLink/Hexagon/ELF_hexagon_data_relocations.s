# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify R_HEX_32 (Pointer32) data relocation.

	.text
	.globl	main
	.type	main,@function
	.p2align	4
main:
	{
		jumpr r31
	}
	.size	main, .-main

	.globl	target_fn
	.type	target_fn,@function
	.p2align	4
target_fn:
	{
		r0 = #42
	}
	{
		jumpr r31
	}
	.size	target_fn, .-target_fn

	.data
	.globl	ptr_to_fn
	.p2align	2
ptr_to_fn:
# jitlink-check: *{4}ptr_to_fn = target_fn
	.word	target_fn
	.size	ptr_to_fn, 4
