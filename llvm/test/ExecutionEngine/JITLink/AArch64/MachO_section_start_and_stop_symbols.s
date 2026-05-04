# RUN: llvm-mc -triple=arm64-apple-darwin24 -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -check %s %t.o

# jitlink-check: *{8}_z = (*{8}_y) + 4

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	2
_main:
        mov     w0, #0
        ret

        .section        __DATA,__custom_section
	.globl	_x
	.p2align	2, 0x0
_x:
	.long	42

	.section	__DATA,__data
	.globl	_y
	.p2align	3, 0x0
_y:
	.quad	section$start$__DATA$__custom_section

	.globl	_z
	.p2align	3, 0x0
_z:
	.quad	section$end$__DATA$__custom_section

.subsections_via_symbols
