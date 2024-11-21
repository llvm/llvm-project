# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=arm64-apple-darwin19 -filetype=obj -o %t/main.o %s
# RUN: llvm-mc -triple=arm64-apple-darwin19 -filetype=obj -o %t/aux_common.o \
# RUN:     %S/Inputs/MachO_common_x_and_addr_getter.s
# RUN: llvm-mc -triple=arm64-apple-darwin19 -filetype=obj -o %t/aux_strong.o \
# RUN:     %S/Inputs/MachO_strong_x_and_addr_getter.s
# RUN: llvm-jitlink -noexec %t/main.o %t/aux_common.o
# RUN: llvm-jitlink -noexec -check=%s %t/main.o %t/aux_strong.o
#
# Check that linking multiple common definitions of the same symbol (in this
# case _x) doesn't lead to an "Unexpected definitions" error.
#
# rdar://132314264
#
# Check that strong defs override:
# jitlink-check: *{8}_x = 42

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	2
_main:
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	bl	_getXAddr
	adrp	x8, _x@GOTPAGE
	ldr	x8, [x8, _x@GOTPAGEOFF]
	cmp	x0, x8
	cset	w0, eq
	ldp	x29, x30, [sp], #16
	ret

	.comm	_x,4,2
.subsections_via_symbols
