# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Combined test exercising constant extenders, duplexes, packets, and
## conditional/compound branches together.  This verifies that the JITLink
## Hexagon backend handles a realistic mix of relocation types in one object.
##
## Relocation types exercised:
##   - R_HEX_B22_PCREL       (plain call)
##   - R_HEX_PLT_B22_PCREL   (PLT call)
##   - R_HEX_B32_PCREL_X     (constant extender for branches)
##   - R_HEX_B15_PCREL_X     (conditional branch, extended)
##   - R_HEX_B9_PCREL_X      (compound compare-jump / duplex jump)
##   - R_HEX_B22_PCREL_X     (extended call)
##   - R_HEX_32_6_X          (constant extender for data)
##   - R_HEX_16_X            (extended immediate load)
##   - R_HEX_8_X             (duplex with extended immediate)
##   - R_HEX_HI16            (high 16 bits)
##   - R_HEX_LO16            (low 16 bits)

	.text
	.globl	main, leaf, helper, plt_call, ext_call, hi_insn, lo_insn
	.type	main,@function
	.p2align	4
main:
	## Extended immediate load: Word32_6_X extender at main.
# jitlink-check: (*{4}main)[13:0] = data_val[19:6]
# jitlink-check: (*{4}main)[27:16] = data_val[31:20]
	{
		r0 = ##data_val
	}
	## Conditional branch with extender.
	{
		if (p0) jump leaf
	}
	## Compound compare-and-jump.
	{
		p0 = cmp.eq(r0, #0)
		if (p0.new) jump:t leaf
	}
	## PLT-style call: resolved as B22_PCREL.
# jitlink-check: (*{4}plt_call)[13:1] = (helper - plt_call)[14:2]
plt_call:
	{
		call helper@PLT
	}
	## Extended call: B32_PCREL_X + B22_PCREL_X.
# jitlink-check: (*{4}ext_call)[13:0] = (helper - ext_call)[19:6]
ext_call:
	{
		call ##helper
	}
	## HI16/LO16 pair for address construction.
# jitlink-check: (*{4}hi_insn)[13:0] = data_val[29:16]
hi_insn:
	{
		r2.h = #hi(data_val)
	}
# jitlink-check: (*{4}lo_insn)[13:0] = data_val[13:0]
lo_insn:
	{
		r2.l = #lo(data_val)
	}
	## Duplex jump (r0 = #0 + jump packs into one duplex word).
	{
		r0 = #0
		jump leaf
	}
	.size	main, .-main

	.type	leaf,@function
	.p2align	4
leaf:
	## Duplex with constant extender: Word32_6_X extender.
# jitlink-check: (*{4}leaf)[13:0] = data_val[19:6]
	{
		r0 = add(r0, ##data_val)
		r1 = #0
	}
	{
		jumpr r31
	}
	.size	leaf, .-leaf

	.type	helper,@function
	.p2align	4
helper:
	## Conditional negated branch.
	{
		if (!p0) jump leaf
	}
	{
		jumpr r31
	}
	.size	helper, .-helper

	.data
	.globl	data_val
	.p2align	2
data_val:
	.word	0xcafebabe
	.size	data_val, 4
