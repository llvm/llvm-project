# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify relocations targeting duplex-encoded instructions.  Duplex
## instructions pack two sub-instructions into a single 32-bit word with
## parse bits [15:14] = 0b00.  The fixup mask lookup functions (findMaskR6,
## findMaskR8, etc.) have a special isDuplex() path returning mask 0x03f00000.
##
## Test 1: { r0 = #0; jump helper } -- duplex jump generates
##         R_HEX_B32_PCREL_X (extender) + R_HEX_B9_PCREL_X (duplex word).
##
## Test 2: { r0 = add(r0, ##data_sym); r1 = #0 } -- duplex with constant
##         extender generates R_HEX_32_6_X (extender) + R_HEX_8_X (duplex
##         word), exercising the isDuplex() path in findMaskR8().

	.text
	.globl	main, helper
	.type	main,@function
	.p2align	4
main:
	## The extender at main uses B32_PCREL_X (mask 0x0fff3fff).
# jitlink-check: (*{4}main)[13:0] = (helper - main)[19:6]
# jitlink-check: (*{4}main)[27:16] = (helper - main)[31:20]
	{
		r0 = #0
		jump helper
	}
	.size	main, .-main

	.type	helper,@function
	.p2align	4
helper:
	## The extender at helper uses Word32_6_X (mask 0x0fff3fff).
# jitlink-check: (*{4}helper)[13:0] = data_sym[19:6]
# jitlink-check: (*{4}helper)[27:16] = data_sym[31:20]
	{
		r0 = add(r0, ##data_sym)
		r1 = #0
	}
	{
		jumpr r31
	}
	.size	helper, .-helper

	.data
	.globl	data_sym
	.p2align	2
data_sym:
	.word	0xdeadbeef
	.size	data_sym, 4
