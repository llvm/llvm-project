# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify R_HEX_B22_PCREL relocation encoding for function calls.
## The B22_PCREL fixup spreads (target - PC) >> 2 into the instruction word
## via mask 0x01ff3ffe: bits [13:1] hold the lower 13 data bits and
## bits [24:16] hold the upper 9 data bits.

	.text
	.globl	main, foo
	.type	main,@function
	.p2align	4
main:
# jitlink-check: (*{4}main)[13:1] = (foo - main)[14:2]
# jitlink-check: (*{4}main)[24:16] = (foo - main)[23:15]
	{
		call foo
	}
	{
		jumpr r31
	}
	.size	main, .-main

	.type	foo,@function
	.p2align	4
foo:
	{
		r0 = #0
	}
	{
		jumpr r31
	}
	.size	foo, .-foo
