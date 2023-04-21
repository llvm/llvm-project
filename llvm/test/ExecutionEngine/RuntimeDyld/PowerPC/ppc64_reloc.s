# test for little endian
#
# This test has been observed to fail on 32-bit architectures (see e.g.
# https://github.com/llvm/llvm-project/issues/62184). Since we're aiming to
# bring up JITLink support for ppc64{le} anyway (see
# https://github.com/llvm/llvm-project/issues/62253) I've opted to just disable
# it on 32-bit architectures for now.
# REQUIRES: llvm-64-bits
#
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o %t/ppc64_reloc.o %s
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o %t/ppc64_elf_module_b.o %S/Inputs/ppc64_elf_module_b.s
# RUN: llvm-rtdyld -triple=powerpc64le-unknown-linux-gnu -verify -check=%s %t/ppc64_reloc.o %t/ppc64_elf_module_b.o
# test for big endian
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=powerpc64-unknown-linux-gnu -filetype=obj -o %t/ppc64_reloc.o %s
# RUN: llvm-mc -triple=powerpc64-unknown-linux-gnu -filetype=obj -o %t/ppc64_elf_module_b.o %S/Inputs/ppc64_elf_module_b.s
# RUN: llvm-rtdyld -triple=powerpc64-unknown-linux-gnu -verify -check=%s %t/ppc64_reloc.o %t/ppc64_elf_module_b.o

	.text
	.abiversion 2
	.file	"test.c"
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
# rtdyld-check: (*{4}call_bl) & 1 = 1
call_bl:
	bl foo
	nop

	li 3, 0
	addi 1, 1, 112
	ld 0, 16(1)
	ld 31, -8(1)
	mtlr 0
# confirm that LK flag is not set for b
# rtdyld-check: (*{4}call_b) & 1 = 0
call_b:
	b foo
	.long	0
	.quad	0
.Lfunc_end0:
	.size	func, .Lfunc_end0-.Lfunc_begin0
