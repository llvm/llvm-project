	.section	__TEXT,__text,regular,pure_instructions
	.globl	_getXAddr
	.p2align	2
_getXAddr:
	adrp	x0, _x@PAGE
	add	x0, x0, _x@PAGEOFF
	ret

	.section	__DATA,__data
	.globl	_x
	.p2align	2, 0x0
_x:
	.long	42

.subsections_via_symbols
