# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o %t/ppc64_reloc.o %s
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o %t/ppc64_elf_module_b.o %S/Inputs/ppc64_elf_module_b.s
# RUN: llvm-jitlink -noexec -check %s %t/ppc64_reloc.o %t/ppc64_elf_module_b.o

	.text
	.abiversion 2
	.file	"test.c"
	.globl main
	.p2align	4
	.type main,@function
main:
	li 3, 0
	blr
	.size main, .-main

	.globl	func
	.p2align	4
	.type	func,@function
func:                                   # @func
.Lfunc_begin0:
.Lfunc_gep0:
	addis 2, 12, .TOC.-.Lfunc_gep0@ha
	addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
	.localentry	func, .Lfunc_lep0-.Lfunc_gep0
	mflr 0
	std 31, -8(1)
	std 0, 16(1)
	stdu 1, -112(1)
	mr 31, 1
# confirm that LK flag is set for bl
# jitlink-check: (*{4}(func+28)) & 1 = 1
call_bl:
	bl foo
	nop
	li 3, 0
	addi 1, 1, 112
	ld 0, 16(1)
	ld 31, -8(1)
	mtlr 0
# confirm that LK flag is not set for b
# jitlink-check: (*{4}(func+56)) & 1 = 0
call_b:
	b foo
	nop
	.long	0
	.quad	0
.Lfunc_end0:
	.size	func, .Lfunc_end0-.Lfunc_begin0
