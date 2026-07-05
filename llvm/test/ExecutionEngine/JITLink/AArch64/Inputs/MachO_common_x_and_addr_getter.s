	.section	__TEXT,__text,regular,pure_instructions
	.globl	_getXAddr
	.p2align	2
_getXAddr:
	adrp	x0, _x@GOTPAGE
	ldr	x0, [x0, _x@GOTPAGEOFF]
	ret

	.comm	_x,4,2
.subsections_via_symbols
