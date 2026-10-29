# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify hardware loop relocation pairs (R_HEX_B32_PCREL_X + R_HEX_B7_PCREL_X).
## The loop0 instruction uses an extender that encodes (target - PC) >> 6.

	.text
	.globl	main, loop_body
	.type	main,@function
	.p2align	4
main:
	## Extender at main: B32_PCREL_X encodes (target - PC) >> 6.
# jitlink-check: (*{4}main)[13:0] = (loop_body - main)[19:6]
# jitlink-check: (*{4}main)[27:16] = (loop_body - main)[31:20]
	{
		loop0(loop_body, #10)
	}
	{
		jumpr r31
	}
	.size	main, .-main

	.type	loop_body,@function
	.p2align	4
loop_body:
	{
		nop
	}:endloop0
	{
		jumpr r31
	}
	.size	loop_body, .-loop_body
