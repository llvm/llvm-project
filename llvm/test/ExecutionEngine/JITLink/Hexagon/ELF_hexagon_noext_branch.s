# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify non-extended branch relocations.  Using an explicit '#' prefix on
## the branch target sets mustNotExtend, preventing assembler relaxation.
## Each branch type spreads (target - PC) >> 2 into non-contiguous bit fields
## via applyMask.  We verify the largest contiguous field run for each type.

	.text
	.globl	main, helper, b13_insn, b9_insn, b7_insn
	.type	main,@function
	.p2align	4
main:
	## B15_PCREL (mask 0x00df20fe): bits [7:1] hold data[6:0].
# jitlink-check: (*{4}main)[7:1] = (helper - main)[8:2]
	{
		if (p0) jump #helper
	}
	## B13_PCREL (mask 0x00202ffe): bits [11:1] hold data[10:0].
# jitlink-check: (*{4}b13_insn)[11:1] = (helper - b13_insn)[12:2]
b13_insn:
	{
		if (r0 == #0) jump:t #helper
	}
	## B9_PCREL (mask 0x003000fe): bits [7:1] hold data[6:0].
# jitlink-check: (*{4}b9_insn)[7:1] = (helper - b9_insn)[8:2]
b9_insn:
	{
		p0 = cmp.eq(r0, #0)
		if (p0.new) jump:t #helper
	}
	## B7_PCREL (mask 0x00001f18): bits [4:3] hold data[1:0],
	## bits [12:8] hold data[6:2].
# jitlink-check: (*{4}b7_insn)[4:3] = (helper - b7_insn)[3:2]
# jitlink-check: (*{4}b7_insn)[12:8] = (helper - b7_insn)[8:4]
b7_insn:
	{
		loop0(#helper, #10)
	}
	{
		jumpr r31
	}
	.size	main, .-main

	.type	helper,@function
	.p2align	4
helper:
	{
		nop
	}:endloop0
	{
		jumpr r31
	}
	.size	helper, .-helper
