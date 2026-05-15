# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify multi-instruction packets with different relocation types.
## Exercises R_HEX_32_6_X + R_HEX_16_X, R_HEX_HI16, R_HEX_LO16,
## and R_HEX_B22_PCREL.

	.text
	.globl	main, helper, hi_insn, lo_insn, call_insn
	.type	main,@function
	.p2align	4
main:
	## Extended immediate load: Word32_6_X extender at main.
# jitlink-check: (*{4}main)[13:0] = data1[19:6]
# jitlink-check: (*{4}main)[27:16] = data1[31:20]
	{
		r0 = ##data1
	}
	## HI16: insn[13:0] = data2[29:16].
# jitlink-check: (*{4}hi_insn)[13:0] = data2[29:16]
hi_insn:
	{
		r1.h = #hi(data2)
	}
	## LO16: insn[13:0] = data2[13:0].
# jitlink-check: (*{4}lo_insn)[13:0] = data2[13:0]
lo_insn:
	{
		r1.l = #lo(data2)
	}
	## B22_PCREL call.
# jitlink-check: (*{4}call_insn)[13:1] = (helper - call_insn)[14:2]
call_insn:
	{
		call helper
	}
	{
		jumpr r31
	}
	.size	main, .-main

	.type	helper,@function
	.p2align	4
helper:
	{
		jumpr r31
	}
	.size	helper, .-helper

	.data
	.globl	data1, data2
	.p2align	2
data1:
	.word	0x11111111
	.size	data1, 4
data2:
	.word	0x22222222
	.size	data2, 4
