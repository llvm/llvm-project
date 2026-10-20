# RUN: llvm-mc -triple=arm64e-apple-macosx -filetype=obj -o %t.o %s
# RUN: llvm-jitlink %t.o
#
# REQUIRES: system-darwin && host=arm64{{.*}}
#
# Check that arm64e ptrauth relocations are handled correctly.
#
# This test contains eight global pointers with different signing schemes
# (IA vs DA key, with and without address diversity, and with 0 or 0xa5a5 as
# the additional diversity value). If all pointers pass authentication at
# runtime then the test returns zero.
#
# This test requires execution since the signed pointers are written by a
# signing function attached to the graph.
#
# TODO: Write an out-of-process version. This will probably need to be added to
# the ORC runtime.

        .section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_main
	.p2align	2
_main:
	adrp	x8, _p1@PAGE
	ldr	x16, [x8, _p1@PAGEOFF]
	autiza	x16

	adrp	x9, _p2@PAGE
	add	x9, x9, _p2@PAGEOFF
	ldr	x16, [x9]
	autia	x16, x9

	adrp	x10, _p3@PAGE
	ldr	x16, [x10, _p3@PAGEOFF]
	mov	x17, #23130
	autia	x16, x17

	adrp	x9, _p4@PAGE
	add	x9, x9, _p4@PAGEOFF
	ldr	x16, [x9]
	mov	x17, x9
	movk	x17, #23130, lsl #48
	autia	x16, x17

	adrp	x10, _p5@PAGE
	ldr	x10, [x10, _p5@PAGEOFF]
	ldraa	x10, [x10]

	adrp	x9, _p6@PAGE
	add	x9, x9, _p6@PAGEOFF
	ldr	x16, [x9]
	autda	x16, x9

	adrp	x10, _p7@PAGE
	ldr	x16, [x10, _p7@PAGEOFF]
	mov	x17, #23130
	autda	x16, x17

	adrp	x9, _p8@PAGE
	add	x9, x9, _p8@PAGEOFF
	ldr	x16, [x9]
	mov	x17, x9
	movk	x17, #23130, lsl #48
	autda	x16, x17

        mov     w0, #0
        ret

	.private_extern	_a
	.section	__DATA,__data
	.globl	_a
	.p2align	3
_a:
	.quad	1

	.private_extern	_b
	.globl	_b
	.p2align	3
_b:
	.quad	2

	.private_extern	_c
	.globl	_c
	.p2align	3
_c:
	.quad	3

	.private_extern	_d
	.globl	_d
	.p2align	3
_d:
	.quad	4

	.private_extern	_e
	.globl	_e
	.p2align	3
_e:
	.quad	5

	.private_extern	_f
	.globl	_f
	.p2align	3
_f:
	.quad	6

	.private_extern	_g
	.globl	_g
	.p2align	3
_g:
	.quad	7

	.private_extern	_h
	.globl	_h
	.p2align	3
_h:
	.quad	8

	.globl	_p1
	.p2align	3
_p1:
	.quad	_a@AUTH(ia,0)

	.globl	_p2
	.p2align	3
_p2:
	.quad	_b@AUTH(ia,0,addr)

	.globl	_p3
	.p2align	3
_p3:
	.quad	_c@AUTH(ia,23130)

	.globl	_p4
	.p2align	3
_p4:
	.quad	_d@AUTH(ia,23130,addr)

	.globl	_p5
	.p2align	3
_p5:
	.quad	_e@AUTH(da,0)

	.globl	_p6
	.p2align	3
_p6:
	.quad	_f@AUTH(da,0,addr)

	.globl	_p7
	.p2align	3
_p7:
	.quad	_g@AUTH(da,23130)

	.globl	_p8y
	.p2align	3
_p8:
	.quad	_h@AUTH(da,23130,addr)

.subsections_via_symbols
