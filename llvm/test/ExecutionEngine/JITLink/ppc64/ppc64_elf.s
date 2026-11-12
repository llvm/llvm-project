# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o %t/ppc64_elf.o %s
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o %t/ppc64_elf_module_b.o %S/Inputs/ppc64_elf_module_b.s
# RUN: llvm-jitlink -noexec -check %s %t/ppc64_elf.o %t/ppc64_elf_module_b.o
	.text
	.abiversion 2
	.file	"Module2.ll"
	.global main
	.p2align 4
	.type main,@function
main:
	li 3, 0
	blr
	.size main, .-main

	.globl	bar                     # -- Begin function bar
	.p2align	4
	.type	bar,@function
.Lfunc_toc0:                            # @bar
	.quad	.TOC.-.Lfunc_gep0
bar:
.Lfunc_begin0:
.Lfunc_gep0:
	ld 2, .Lfunc_toc0-.Lfunc_gep0(12)
	add 2, 2, 12
.Lfunc_lep0:
	.localentry	bar, .Lfunc_lep0-.Lfunc_gep0
# %bb.0:
	mflr 0
	std 0, 16(1)
	stdu 1, -32(1)
# jitlink-check: (*{8}got_addr(ppc64_elf.o, foo)) = foo_gep
# jitlink-check: decode_operand(bar+20, 0) = (stub_addr(ppc64_elf.o, foo) - (bar+20)) >> 2
foo_call:
	bl foo
	nop
	addi 1, 1, 32
	ld 0, 16(1)
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	bar, .Lfunc_end0-.Lfunc_begin0
                                        # -- End function

	.section	".note.GNU-stack","",@progbits
