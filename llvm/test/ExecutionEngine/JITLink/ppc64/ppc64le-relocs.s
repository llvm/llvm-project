# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-jitlink -abs external_var=0xffff0000 -abs puts=0xffff6400 -abs \
# RUN:   foo=0xffff8800 -noexec %t
#
# Check typical relocations involving external function call, external variable
# reference, local function call and referencing global variable defined in the
# same CU. This test serves as smoke test, `llvm-jitlink -check` is not used.

	.text
	.abiversion 2
	.file	"ppc64le-relocs.c"
	.globl	main
	.p2align	4
	.type	main,@function
main:
.Lfunc_begin0:
	li 3, 0
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	main, .Lfunc_end0-.Lfunc_begin0

	.globl	id
	.p2align	4
	.type	id,@function
id:
.Lfunc_begin1:
.Lfunc_gep1:
	addis 2, 12, .TOC.-.Lfunc_gep1@ha
	addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep1:
	.localentry	id, .Lfunc_lep1-.Lfunc_gep1
	addis 4, 2, .LC0@toc@ha
	ld 4, .LC0@toc@l(4)
	lwz 4, 0(4)
	sub	3, 4, 3
	extsw 3, 3
	blr
	.long	0
	.quad	0
.Lfunc_end1:
	.size	id, .Lfunc_end1-.Lfunc_begin1

# Test referencing external data via R_PPC64_TOC16HA and R_PPC64_TOC16LO.
	.globl	test_reference_external_data
	.p2align	4
	.type	test_reference_external_data,@function
test_reference_external_data:
.Lfunc_begin2:
.Lfunc_gep2:
	addis 2, 12, .TOC.-.Lfunc_gep2@ha
	addi 2, 2, .TOC.-.Lfunc_gep2@l
.Lfunc_lep2:
	.localentry	test_reference_external_data, .Lfunc_lep2-.Lfunc_gep2
	addis 3, 2, .LC0@toc@ha
	ld 3, .LC0@toc@l(3)
	blr
	.long	0
	.quad	0
.Lfunc_end2:
	.size	test_reference_external_data, .Lfunc_end2-.Lfunc_begin2

# Test referencing global variable defined in the same CU.
	.globl	test_reference_local_data
	.p2align	4
	.type	test_reference_local_data,@function
test_reference_local_data:
.Lfunc_begin3:
.Lfunc_gep3:
	addis 2, 12, .TOC.-.Lfunc_gep3@ha
	addi 2, 2, .TOC.-.Lfunc_gep3@l
.Lfunc_lep3:
	.localentry	test_reference_local_data, .Lfunc_lep3-.Lfunc_gep3
	addis 3, 2, .LC1@toc@ha
	ld 3, .LC1@toc@l(3)
	blr
	.long	0
	.quad	0
.Lfunc_end3:
	.size	test_reference_local_data, .Lfunc_end3-.Lfunc_begin3

# Test external function call with R_PPC64_REL24, which requires PLT
# call stub.
	.globl	test_external_call
	.p2align	4
	.type	test_external_call,@function
test_external_call:
.Lfunc_begin4:
.Lfunc_gep4:
	addis 2, 12, .TOC.-.Lfunc_gep4@ha
	addi 2, 2, .TOC.-.Lfunc_gep4@l
.Lfunc_lep4:
	.localentry	test_external_call, .Lfunc_lep4-.Lfunc_gep4
	mflr 0
	stdu 1, -32(1)
	addis 3, 2, .L.str@toc@ha
	std 0, 48(1)
	addi 3, 3, .L.str@toc@l
	bl puts
	nop
	addi 1, 1, 32
	ld 0, 16(1)
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end4:
	.size	test_external_call, .Lfunc_end4-.Lfunc_begin4

# Test local calls with R_PPC64_REL24.
# Calling to `id` has a nop followed, while there is no
# nop after calling `id1`.
	.globl	test_local_call
	.p2align	4
	.type	test_local_call,@function
test_local_call:
.Lfunc_begin5:
.Lfunc_gep5:
	addis 2, 12, .TOC.-.Lfunc_gep5@ha
	addi 2, 2, .TOC.-.Lfunc_gep5@l
.Lfunc_lep5:
	.localentry	test_local_call, .Lfunc_lep5-.Lfunc_gep5
	mflr 0
	std 29, -24(1)
	std 30, -16(1)
	stdu 1, -64(1)
	std 0, 80(1)
	mr	30, 3
# A local call, with a nop followed.
	bl id
	nop
	mr	29, 3
	mr	3, 30
# A local call, without nop followed.
	bl id1
	add 3, 3, 29
	extsw 3, 3
	addi 1, 1, 64
	ld 0, 16(1)
	ld 30, -16(1)
	ld 29, -24(1)
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end5:
	.size	test_local_call, .Lfunc_end5-.Lfunc_begin5

	.p2align	4
	.type	id1,@function
id1:
.Lfunc_begin6:
.Lfunc_gep6:
	addis 2, 12, .TOC.-.Lfunc_gep6@ha
	addi 2, 2, .TOC.-.Lfunc_gep6@l
.Lfunc_lep6:
	.localentry	id1, .Lfunc_lep6-.Lfunc_gep6
	addis 4, 2, .LC1@toc@ha
	ld 4, .LC1@toc@l(4)
	lwz 4, 0(4)
	sub	3, 4, 3
	extsw 3, 3
	blr
	.long	0
	.quad	0
.Lfunc_end6:
	.size	id1, .Lfunc_end6-.Lfunc_begin6

# Test external function call with R_PPC64_REL24_NOTOC, which requires PLT
# call stub, however no saving of r2 is required and there's no nop after
# the branch instruction.
	.globl	bar
	.p2align	4
	.type	bar,@function
bar:
.Lfunc_begin7:
	.localentry	bar, 1
	b foo@notoc
	#TC_RETURNd8 foo@notoc 0
	.long	0
	.quad	0
.Lfunc_end7:
	.size	bar, .Lfunc_end7-.Lfunc_begin7

	.type	local_var,@object
	.section	.bss,"aw",@nobits
	.globl	local_var
	.p2align	2, 0x0
local_var:
	.long	0
	.size	local_var, 4

	.type	.L.str,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Hey!"
	.size	.L.str, 5

	.section	.toc,"aw",@progbits
.LC0:
	.tc external_var[TC],external_var
.LC1:
	.tc local_var[TC],local_var
