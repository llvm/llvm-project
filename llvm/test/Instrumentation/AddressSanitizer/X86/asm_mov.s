# RUN: llvm-mc %s -triple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+sse2 -asm-instrumentation=address -asan-instrument-assembly | FileCheck %s

	.text
	.globl	mov1b
	.align	16, 0x90
	.type	mov1b,@function
# CHECK-LABEL: mov1b:
#
# CHECK: leaq -128(%rsp), %rsp
# CHECK: callq __asan_report_load1@PLT
# CHECK: leaq 128(%rsp), %rsp
#
# CHECK: movb (%rsi), %al
#
# CHECK: leaq -128(%rsp), %rsp
# CHECK: callq __asan_report_store1@PLT
# CHECK: leaq 128(%rsp), %rsp
#
# CHECK: movb %al, (%rdi)
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

	.globl	mov16b
	.align	16, 0x90
	.type	mov16b,@function
# CHECK-LABEL: mov16b:
#
# CHECK: leaq -128(%rsp), %rsp
# CHECK: callq __asan_report_load16@PLT
# CHECK: leaq 128(%rsp), %rsp
#
# CHECK: movaps (%rsi), %xmm0
#
# CHECK: leaq -128(%rsp), %rsp
# CHECK: callq __asan_report_store16@PLT
# CHECK: leaq 128(%rsp), %rsp
#
# CHECK: movaps %xmm0, (%rdi)
mov16b:                                 # @mov16b
	.cfi_startproc
# %bb.0:
	#APP
	movaps	(%rsi), %xmm0
	movaps	%xmm0, (%rdi)

	#NO_APP
	retq
.Ltmp1:
	.size	mov16b, .Ltmp1-mov16b
	.cfi_endproc


	.ident	"clang version 3.5 "
	.section	".note.GNU-stack","",@progbits
