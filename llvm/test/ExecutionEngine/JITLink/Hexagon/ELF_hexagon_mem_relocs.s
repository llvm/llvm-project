# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify memory-operation and data relocations that exercise different
## fixup mask widths:
##   - R_HEX_32_6_X + R_HEX_6_X    (memop with extended offset, 6-bit field)
##   - R_HEX_32_6_X + R_HEX_11_X   (memb load with extended offset, 11-bit field)
##   - R_HEX_32_6_X + R_HEX_12_X   (conditional transfer, 12-bit field)
##   - R_HEX_32_6_X + R_HEX_9_X    (unsigned compare, 9-bit field)
##   - R_HEX_32_6_X + R_HEX_10_X   (signed compare, 10-bit field)
##   - R_HEX_B32_PCREL_X + R_HEX_6_PCREL_X  (PC-relative add, 6-bit field)
##   - R_HEX_32_PCREL               (PC-relative data reference)
##   - R_HEX_GD_PLT_B22_PCREL      (GD_PLT call, mapped to plain branch)

	.text
	.globl	main, helper, memb_ext, cond_ext, pcrel_ext, gdplt_call
	.type	main,@function
	.p2align	4
main:
	## Memop with extended offset: generates R_HEX_32_6_X + R_HEX_6_X.
	## Verify Word32_6_X extender encoding (mask 0x0fff3fff).
# jitlink-check: (*{4}main)[13:0] = data_val[19:6]
# jitlink-check: (*{4}main)[27:16] = data_val[31:20]
	{
		memw(r0+##data_val) += r1
	}
	## Byte load with extended offset: generates R_HEX_32_6_X + R_HEX_11_X.
# jitlink-check: (*{4}memb_ext)[13:0] = data_val[19:6]
memb_ext:
	{
		r0 = memb(r1+##data_val)
	}
	## Conditional transfer: generates R_HEX_32_6_X + R_HEX_12_X.
# jitlink-check: (*{4}cond_ext)[13:0] = data_val[19:6]
cond_ext:
	{
		if (p0) r0 = ##data_val
	}
	## Unsigned compare: generates R_HEX_32_6_X + R_HEX_9_X.
	{
		p0 = !cmp.gtu(r0, ##data_val)
	}
	## Signed compare: generates R_HEX_32_6_X + R_HEX_10_X.
	{
		p0 = !cmp.gt(r0, ##data_val)
	}
	## PC-relative add: generates R_HEX_B32_PCREL_X + R_HEX_6_PCREL_X.
	## Verify B32_PCREL_X extender encoding.
# jitlink-check: (*{4}pcrel_ext)[13:0] = (data_val - pcrel_ext)[19:6]
# jitlink-check: (*{4}pcrel_ext)[27:16] = (data_val - pcrel_ext)[31:20]
pcrel_ext:
	{
		r0 = add(pc, ##data_val@PCREL)
	}
	## GD_PLT call: resolved as B22_PCREL.
# jitlink-check: (*{4}gdplt_call)[13:1] = (helper - gdplt_call)[14:2]
gdplt_call:
	{
		call helper@GDPLT
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
	.globl	data_val
	.p2align	2
data_val:
	.word	0x42
	.size	data_val, 4

## PC-relative data reference: generates R_HEX_32_PCREL.
	.globl	pcrel_ref
	.p2align	2
pcrel_ref:
# jitlink-check: *{4}pcrel_ref = helper - pcrel_ref
	.word	helper - pcrel_ref
	.size	pcrel_ref, 4
