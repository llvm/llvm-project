# This module contains a function with its local and global entry points
# exposed. It is used by the ppc64_elf test to verify that functions with
# different TOCs are called via their global entry points.
	.text
	.abiversion 2
	.file	"ppc64_elf_module_b.ll"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2               # -- Begin function foo
.LCPI0_0:
	.long	1093664768              # float 11
	.text
	.globl	foo
	.p2align	4
	.type	foo,@function
.Lfunc_toc0:                            # @foo
	.quad	.TOC.-foo_gep
foo:
.Lfunc_begin0:
	.cfi_startproc
        .globl  foo_gep
foo_gep:
	ld 2, .Lfunc_toc0-foo_gep(12)
	add 2, 2, 12
        .globl  foo_lep
foo_lep:
	.localentry	foo, foo_lep-foo_gep
# %bb.0:
	addis 3, 2, .LC0@toc@ha
	ld 3, .LC0@toc@l(3)
	lfsx 1, 0, 3
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	foo, .Lfunc_end0-.Lfunc_begin0
	.cfi_endproc
                                        # -- End function
	.section	.toc,"aw",@progbits
.LC0:
	.tc .LCPI0_0[TC],.LCPI0_0

	.section	".note.GNU-stack","",@progbits
