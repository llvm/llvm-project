	#RUN: llvm-mc -preserve-comments -n -triple i386-linux-gnu < %s > %t
	.text
foo:
	nop #