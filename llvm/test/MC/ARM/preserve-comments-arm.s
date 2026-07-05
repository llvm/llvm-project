	@RUN: llvm-mc -preserve-comments -n -triple arm-eabi < %s > %t
	@RUN: sed 's/#[C]omment/@Comment/g' %s > %t2
	@RUN: diff -b %t %t2
	.text

	mov	r0, r0
foo:	#Comment here
	mov	r0, r0	@ EOL comment
	.ident	""

