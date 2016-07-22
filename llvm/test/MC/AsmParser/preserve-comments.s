	#RUN: llvm-mc -preserve-comments -n -triple i386-linux-gnu < %s > %t
	#RUN: diff %s %t
	.text

foo:	#Comment here
	#comment here
	nop
	#if DIRECTIVE COMMENT
	## WHOLE LINE COMMENT
	cmpl	$196, %eax	## EOL COMMENT
	#endif
	.ident	"clang version 3.9.0"
	.section	".note.GNU-stack","",@progbits
