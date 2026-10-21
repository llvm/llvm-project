# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify compound compare-and-jump relocation pairs (R_HEX_B32_PCREL_X +
## R_HEX_B9_PCREL_X).  The compound instruction includes an extender that
## encodes (target - PC) >> 6.

	.text
	.globl	main, helper
	.type	main,@function
	.p2align	4
main:
	## Extender at main: B32_PCREL_X encodes (target - PC) >> 6.
# jitlink-check: (*{4}main)[13:0] = (helper - main)[19:6]
# jitlink-check: (*{4}main)[27:16] = (helper - main)[31:20]
	{
		p0 = cmp.eq(r0, #0)
		if (p0.new) jump:t helper
	}
	{
		p0 = cmp.gt(r0, #0)
		if (!p0.new) jump:t helper
	}
	{
		jumpr r31
	}
	.size	main, .-main

	.type	helper,@function
	.p2align	4
helper:
	{
		r0 = #0
	}
	{
		jumpr r31
	}
	.size	helper, .-helper
