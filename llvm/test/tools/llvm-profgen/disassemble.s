# REQUIRES: x86-registered-target
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-profgen --binary=%t --perfscript=%s --output=%t1 -show-disassembly -x86-asm-syntax=intel | FileCheck %s --match-full-lines

# CHECK: Disassembly of section .text [0x0, 0x66]:
# CHECK: <foo1>:
# CHECK:        0:	push	rbp
# CHECK:        1:	mov	rbp, rsp
# CHECK:        4:	sub	rsp, 16
# CHECK:        8:	mov	dword ptr [rbp - 4], 0
# CHECK:        f:	mov	edi, 1
# CHECK:       14:	call	0x19
# CHECK:       19:	mov	edi, 2
# CHECK:       1e:	mov	dword ptr [rbp - 8], eax
# CHECK:       21:	call	0x26
# CHECK:       26:	mov	ecx, dword ptr [rbp - 8]
# CHECK:       29:	add	ecx, eax
# CHECK:       2b:	mov	eax, ecx
# CHECK:       2d:	add	rsp, 16
# CHECK:       31:	pop	rbp
# CHECK:       32:	ret

# CHECK: <foo2>:
# CHECK:       33:	push	rbp
# CHECK:       34:	mov	rbp, rsp
# CHECK:       37:	sub	rsp, 16
# CHECK:       3b:	mov	dword ptr [rbp - 4], 0
# CHECK:       42:	mov	edi, 1
# CHECK:       47:	call	0x4c
# CHECK:       4c:	mov	edi, 2
# CHECK:       51:	mov	dword ptr [rbp - 8], eax
# CHECK:       54:	call	0x59
# CHECK:       59:	mov	ecx, dword ptr [rbp - 8]
# CHECK:       5c:	add	ecx, eax
# CHECK:       5e:	mov	eax, ecx
# CHECK:       60:	add	rsp, 16
# CHECK:       64:	pop	rbp
# CHECK:       65:	ret



.section .text
foo1:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$16, %rsp
	movl	$0, -4(%rbp)
	movl	$1, %edi
	callq	_Z5funcAi
	movl	$2, %edi
	movl	%eax, -8(%rbp)
	callq	_Z5funcBi
	movl	-8(%rbp), %ecx
	addl	%eax, %ecx
	movl	%ecx, %eax
	addq	$16, %rsp
	popq	%rbp
	retq

.section .text
foo2:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$16, %rsp
	movl	$0, -4(%rbp)
	movl	$1, %edi
	callq	_Z5funcBi
	movl	$2, %edi
	movl	%eax, -8(%rbp)
	callq	_Z5funcAi
	movl	-8(%rbp), %ecx
	addl	%eax, %ecx
	movl	%ecx, %eax
	addq	$16, %rsp
	popq	%rbp
	retq

# CHECK: Disassembly of section .text.hot [0x0, 0x12]:
# CHECK: <bar>:
# CHECK:        0:	push	rbp
# CHECK:        1:	mov	rbp, rsp
# CHECK:        4:	mov	dword ptr [rbp - 4], edi
# CHECK:        7:	mov	dword ptr [rbp - 8], esi
# CHECK:        a:	mov	eax, dword ptr [rbp - 4]
# CHECK:        d:	add	eax, dword ptr [rbp - 8]
# CHECK:       10:	pop	rbp
# CHECK:       11:	ret

.section .text.hot
bar:
	pushq	%rbp
	movq	%rsp, %rbp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-4(%rbp), %eax
	addl	-8(%rbp), %eax
	popq	%rbp
	retq


# CHECK: Disassembly of section .text.unlikely [0x0, 0x12]:
# CHECK: <baz>:
# CHECK:        0:	push	rbp
# CHECK:        1:	mov	rbp, rsp
# CHECK:        4:	mov	dword ptr [rbp - 4], edi
# CHECK:        7:	mov	dword ptr [rbp - 8], esi
# CHECK:        a:	mov	eax, dword ptr [rbp - 4]
# CHECK:        d:	sub	eax, dword ptr [rbp - 8]
# CHECK:       10:	pop	rbp
# CHECK:       11:	ret

.section .text.unlikely
baz:
	pushq	%rbp
	movq	%rsp, %rbp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-4(%rbp), %eax
	subl	-8(%rbp), %eax
	popq	%rbp
	retq
