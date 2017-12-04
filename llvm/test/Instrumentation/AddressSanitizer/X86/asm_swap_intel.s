# RUN: llvm-mc %s -x86-asm-syntax=intel -triple=x86_64-unknown-linux-gnu -asm-instrumentation=address -asan-instrument-assembly | FileCheck %s

	.text
	.globl	swap
	.align	16, 0x90
	.type	swap,@function
# CHECK-LABEL: swap:
#
# CHECK: leaq -128(%rsp), %rsp
# CHECK: callq __asan_report_load8@PLT
# CHECK: leaq 128(%rsp), %rsp
#
# CHECK: movq (%rcx), %rax
#
# CHECK: leaq -128(%rsp), %rsp
# CHECK: callq __asan_report_load8@PLT
# CHECK: leaq 128(%rsp), %rsp
#
# CHECK: movq (%rdx), %rbx
#
# CHECK: leaq -128(%rsp), %rsp
# CHECK: callq __asan_report_store8@PLT
# CHECK: leaq 128(%rsp), %rsp
#
# CHECK: movq %rbx, (%rcx)
#
# CHECK: leaq -128(%rsp), %rsp
# CHECK: callq __asan_report_store8@PLT
# CHECK: leaq 128(%rsp), %rsp
#
# CHECK: movq %rax, (%rdx)
swap:                                   # @swap
	.cfi_startproc
# %bb.0:
	push	rbx
.Ltmp0:
	.cfi_def_cfa_offset 16
.Ltmp1:
	.cfi_offset rbx, -16
	mov	rcx, rdi
	mov	rdx, rsi
	#APP


	mov	rax, qword ptr [rcx]
	mov	rbx, qword ptr [rdx]
	mov	qword ptr [rcx], rbx
	mov	qword ptr [rdx], rax

	#NO_APP
	pop	rbx
	ret
.Ltmp2:
	.size	swap, .Ltmp2-swap
	.cfi_endproc


	.ident	"clang version 3.5.0 "
	.section	".note.GNU-stack","",@progbits
