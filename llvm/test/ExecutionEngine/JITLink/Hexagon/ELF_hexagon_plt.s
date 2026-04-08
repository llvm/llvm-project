# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify that PLT-style relocations link correctly.  JITLink resolves all
## symbols directly, so PLT and GD_PLT variants are mapped to plain branch
## edges.  This test covers:
##   - R_HEX_PLT_B22_PCREL (call @PLT)
##   - R_HEX_GD_PLT_B32_PCREL_X + R_HEX_GD_PLT_B22_PCREL_X (call ##@GDPLT)

	.text
	.globl	main, helper, gdplt_call
	.type	main,@function
	.p2align	4
main:
	## PLT call: resolved as B22_PCREL (mask 0x01ff3ffe).
# jitlink-check: (*{4}main)[13:1] = (helper - main)[14:2]
# jitlink-check: (*{4}main)[24:16] = (helper - main)[23:15]
	{
		call helper@PLT
	}
	## Extended GD_PLT call: resolved as B32_PCREL_X + B22_PCREL_X.
# jitlink-check: (*{4}gdplt_call)[13:0] = (helper - gdplt_call)[19:6]
# jitlink-check: (*{4}gdplt_call)[27:16] = (helper - gdplt_call)[31:20]
gdplt_call:
	{
		call ##helper@GDPLT
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
