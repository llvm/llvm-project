@ RUN: llvm-mc -triple arm %s | FileCheck %s

	.data

short:
	.short 0
	.short 0xdefe

@ CHECK-LABEL: short
@ CHECK-NEXT:	.short 0
@ CHECK-NEXT:	.short 57086

hword:
	.hword 0
	.hword 0xdefe

@ CHECK-LABEL: hword
@ CHECK-NEXT:	.short 0
@ CHECK-NEXT:	.short 57086

word:
	.word 3

@ CHECK-LABEL: word
@ CHECK-NEXT:	.long 3

