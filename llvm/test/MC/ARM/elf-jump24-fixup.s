@ RUN: llvm-mc %s -triple=thumbv7-linux-gnueabi -filetype=obj -o - < %s | llvm-objdump -r - | FileCheck %s
	.syntax unified
	.text
	.code	16
	.thumb_func
foo:
	b.w	bar

@ CHECK: {{[0-9a-f]+}} R_ARM_THM_JUMP24 bar
