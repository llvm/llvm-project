	.text
	.abiversion 2
	.weak	foo
	.p2align	4
	.type	foo,@function
foo:
.Lfunc_begin0:
	li 3, 1
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	foo, .Lfunc_end0-.Lfunc_begin0
