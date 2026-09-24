# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify R_HEX_HI16 and R_HEX_LO16 relocation encoding.
## Both use mask 0x00c03fff: bits [13:0] hold the lower 14 data bits and
## bits [23:22] hold the upper 2 data bits.
## HI16 encodes target[31:16]; LO16 encodes target[15:0].

	.text
	.globl	main, target, lo_insn
	.type	main,@function
	.p2align	4
main:
## HI16: insn[13:0] = target[29:16], insn[23:22] = target[31:30].
# jitlink-check: (*{4}main)[13:0] = target[29:16]
# jitlink-check: (*{4}main)[23:22] = target[31:30]
	{
		r0.h = #hi(target)
	}
## LO16: insn[13:0] = target[13:0], insn[23:22] = target[15:14].
# jitlink-check: (*{4}lo_insn)[13:0] = target[13:0]
# jitlink-check: (*{4}lo_insn)[23:22] = target[15:14]
lo_insn:
	{
		r0.l = #lo(target)
	}
	{
		jumpr r31
	}
	.size	main, .-main

	.type	target,@function
	.p2align	4
target:
	{
		r0 = #42
	}
	{
		jumpr r31
	}
	.size	target, .-target
