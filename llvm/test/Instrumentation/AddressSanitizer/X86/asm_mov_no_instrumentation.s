# RUN: llvm-mc %s -triple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+sse2 | FileCheck %s

	.text
	.globl	mov1b
	.align	16, 0x90
	.type	mov1b,@function
# CHECK-LABEL: mov1b
# CHECK: movb (%rsi), %al
# CHECK: movb %al, (%rdi)
# CHECK-NOT: callq __asan_report_load1@PLT
# CHECK-NOT: callq __asan_report_store1@PLT
mov1b:                                  # @mov1b
	.cfi_startproc
# %bb.0:
	#APP
	movb	(%rsi), %al
	movb	%al, (%rdi)

	#NO_APP
	retq
.Ltmp0:
	.size	mov1b, .Ltmp0-mov1b
	.cfi_endproc

	.ident	"clang version 3.5 "
	.section	".note.GNU-stack","",@progbits
