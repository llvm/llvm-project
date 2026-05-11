# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x10000 -slab-allocate 64Kb \
# RUN:   -slab-page-size 4096 -check %s %t.o

## Verify constant extender relocation pairs.
## Each extended instruction is two words: an immext (constant extender) at
## the label address, followed by the main instruction at label + 4.
##
## B32_PCREL_X (mask 0x0fff3fff) encodes (target - PC) >> 6.
## B22_PCREL_X (mask 0x01ff3ffe) encodes low 6 bits (assembler adds +4 addend).
## Word32_6_X  (mask 0x0fff3fff) encodes target >> 6.

	.text
	.globl	main, far_func, data_ext
	.type	main,@function
	.p2align	4
main:
	## Extended call: immext (B32_PCREL_X) at main, call (B22_PCREL_X) at main+4.
# jitlink-check: (*{4}main)[13:0] = (far_func - main)[19:6]
# jitlink-check: (*{4}main)[27:16] = (far_func - main)[31:20]
# jitlink-check: (*{4}(main + 4))[6:1] = (far_func - main)[5:0]
	{
		call ##far_func
	}
	## Extended load: immext (Word32_6_X) at data_ext, main (Word16_X) at data_ext+4.
# jitlink-check: (*{4}data_ext)[13:0] = far_data[19:6]
# jitlink-check: (*{4}data_ext)[27:16] = far_data[31:20]
data_ext:
	{
		r0 = ##far_data
	}
	{
		jumpr r31
	}
	.size	main, .-main

	.type	far_func,@function
	.p2align	4
far_func:
	{
		jumpr r31
	}
	.size	far_func, .-far_func

	.data
	.globl	far_data
	.p2align	2
far_data:
	.word	0x12345678
	.size	far_data, 4
