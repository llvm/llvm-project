# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify conditional branch relocation pairs (R_HEX_B32_PCREL_X +
## R_HEX_B15_PCREL_X).  The assembler always generates extended forms for
## conditional jumps: immext (B32_PCREL_X) at the label, branch (B15_PCREL_X)
## at label + 4.

	.text
	.globl	main, helper
	.type	main,@function
	.p2align	4
main:
	## Extender: B32_PCREL_X encodes (target - PC) >> 6.
# jitlink-check: (*{4}main)[13:0] = (helper - main)[19:6]
# jitlink-check: (*{4}main)[27:16] = (helper - main)[31:20]
	## Branch: B15_PCREL_X encodes low 6 bits in bits [6:1] (addend +4).
# jitlink-check: (*{4}(main + 4))[6:1] = (helper - main)[5:0]
	{
		if (p0) jump helper
	}
	{
		if (!p0) jump helper
	}
	{
		jumpr r31
	}
	.size	main, .-main

	.type	helper,@function
	.p2align	4
helper:
	{
		r0 = #42
	}
	{
		jumpr r31
	}
	.size	helper, .-helper
