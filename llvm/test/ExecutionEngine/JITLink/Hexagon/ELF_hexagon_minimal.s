# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify that a minimal Hexagon ELF object links without error and that the
## instruction data is preserved (no relocations -- just a raw encoding check).

	.text
	.globl	main
	.type	main,@function
	.p2align	4
main:
# jitlink-check: *{4}main = 0x529fc000
	{
		jumpr r31
	}
	.size	main, .-main
