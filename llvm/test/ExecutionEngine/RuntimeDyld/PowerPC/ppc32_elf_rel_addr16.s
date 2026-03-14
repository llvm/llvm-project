# RUN: llvm-mc -triple=powerpc-unknown-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-rtdyld -triple=powerpc-unknown-linux-gnu -verify -check=%s %t
	.text
	.file	"ppc32_elf_rel_addr16.ll"
	.globl	lookup
	.align	2
	.type	lookup,@function
lookup:                                 # @lookup
.Lfunc_begin0:
# %bb.0:
	stw 31, -4(1)
	stwu 1, -16(1)
insn_hi:
# Check the higher 16-bits of the symbol's absolute address
# rtdyld-check: decode_operand(insn_hi, 1) = elements[31:16]
	lis 4, elements@ha
	slwi 3, 3, 2
	mr 31, 1
insn_lo:
# Check the lower 16-bits of the symbol's absolute address
# rtdyld-check: decode_operand(insn_lo, 2) = elements[15:0]
	la 4, elements@l(4)
	lwzx 3, 4, 3
	addi 1, 1, 16
	lwz 31, -4(1)
	blr
.Lfunc_end0:
	.size	lookup, .Lfunc_end0-.Lfunc_begin0

	.type	elements,@object        # @elements
	.data
	.globl	elements
	.align	2
elements:
	.long	14                      # 0xe
	.long	4                       # 0x4
	.long	1                       # 0x1
	.long	3                       # 0x3
	.long	13                      # 0xd
	.long	0                       # 0x0
	.long	32                      # 0x20
	.long	334                     # 0x14e
	.size	elements, 32


	.ident	"clang version 3.7.0 "
	.section	".note.GNU-stack","",@progbits
