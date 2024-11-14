## A larger branch relaxation test that requires multiple relaxation steps.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

## Regression test that we use jump short instead of jump near
## (see the reverted be5a845e4c29aadb513ae6e5e2879dccf37efdbb).
# CHECK:      4a9: eb 00                         jmp     0x4ab
# CHECK:      5a3: eb 00                         jmp     0x5a5
# CHECK-NEXT: 5a5: eb 00                         jmp     0x5a7
# CHECK-NEXT: 5a7: eb 00                         jmp     0x5a9
# CHECK-NEXT: 5a9: eb 00                         jmp     0x5ab
# CHECK-NEXT: 5ab: e9 fb fe ff ff                jmp     0x4ab
# CHECK-NEXT: 5b0: eb 00                         jmp     0x5b2

	pushq	%rbp
	movq	%rsp, %rbp
	movq	%rcx, (%rbp)
	movq	%r8, (%rbp)
	movl	$0, (%rbp)
	movl	$0, (%rbp)
	movl	$0, (%rbp)
	cmpq	$0, (%rbp)
	je	.LBB0_2
	movl	$0, (%rbp)
	jmp	.LBB0_30
.LBB0_2:
	jmp	.LBB0_3
.LBB0_3:
	movq	(%rbp), %rax
	movsbl	(%rax), %ecx
	xorl	%eax, %eax
	cmpl	%ecx, %eax
	jne	.LBB0_29
	jmp	.LBB0_5
.LBB0_5:
	movb	$0, %al
	callq	sqlite3_stepPLT
	movl	%eax, (%rbp)
	cmpq	$0, (%rbp)
	je	.LBB0_23
	movl	$100, %eax
	cmpl	(%rbp), %eax
	sete	%al
	andb	$1, %al
	movzbl	%al, %eax
	cmpl	(%rbp), %eax
	jne	.LBB0_23
	cmpl	$0, (%rbp)
	je	.LBB0_23
	movq	(%rbp), %rax
	movl	(%rax), %eax
	andl	$256, %eax
	cmpl	$0, %eax
	je	.LBB0_23
	cmpl	$0, (%rbp)
	je	.LBB0_22
	movl	$0, (%rbp)
.LBB0_11:
	cmpl	$0, (%rbp)
	je	.LBB0_13
	movl	(%rbp), %esi
	xorl	%edi, %edi
	movb	$0, %al
	callq	sqlite3_free_tablePLT
	jmp	.LBB0_11
.LBB0_13:
	movq	(%rbp), %rax
	movslq	(%rbp), %rcx
	cmpb	$0, (%rax,%rcx)
	je	.LBB0_15
	movq	(%rbp), %rax
	movb	$1, 8(%rax)
.LBB0_15:
	movq	(%rbp), %rax
	movslq	(%rbp), %rcx
	addq	%rcx, %rax
	movq	%rax, (%rip)
	movl	$0, (%rbp)
.LBB0_16:
	movl	(%rbp), %eax
	cmpl	(%rbp), %eax
	jge	.LBB0_21
	movl	$0, (%rbp)
	movl	$0, (%rbp)
	movq	(%rbp), %rdi
	movb	$0, %al
	callq	sqlite3_free_tablePLT
	movq	(%rip), %rax
	movslq	(%rbp), %rcx
	movl	$5, %edx
	cmpq	%rdx, (%rax,%rcx,)
	je	.LBB0_19
	movq	(%rbp), %rax
	movb	$1, 8(%rax)
	jmp	.LBB0_30
.LBB0_19:
	jmp	.LBB0_20
.LBB0_20:
	movl	(%rbp), %eax
	addl	$1, %eax
	movl	%eax, (%rbp)
	jmp	.LBB0_16
.LBB0_21:
	jmp	.LBB0_22
.LBB0_22:
	jmp	.LBB0_23
.LBB0_23:
	cmpl	$0, (%rbp)
	je	.LBB0_27
	movb	$0, %al
	callq	sqlite3_stepPLT
	movl	%eax, (%rbp)
	movl	$0, (%rbp)
	cmpl	$0, (%rbp)
	je	.LBB0_26
	jmp	.LBB0_28
.LBB0_26:
	jmp	.LBB0_27
.LBB0_27:
	jmp	.LBB0_5
.LBB0_28:
	jmp	.LBB0_3
.LBB0_29:
	jmp	.LBB0_30
.LBB0_30:
	movl	(%rbp), %eax
	addq	$96, %rsp
	popq	%rbp
	retq
	.p2align	4, 0x90
	movq	%rsp, %rbp
	subq	$48, %rsp
	movq	%rdi, (%rbp)
	movl	%esi, (%rbp)
	movl	%edx, (%rbp)
	cmpq	$0, sqlite3_get_table_res_00
	movq	(%rbp), %rax
	jmp	.LBB1_8
	movq	$0, (%rbp)
	movq	(%rbp), %rdi
	leaq	sqlite3_get_table_zSql0, %rsi
	leaq	sqlite3_get_table_cb0, %rdx
	leaq	sqlite3_get_table_res0, %rcx
	leaq	(%rip), %r8
	callq	sqlite3_exec
	movl	%eax, (%rip)
	movslq	(%rbp), %rax
	movq	%rax, (%rbp)
	movl	(%rip), %eax
	andl	$5, %eax
	cmpl	$0, %eax
	je	.LBB1_4
	movl	(%rip), %ecx
	movq	(%rbp), %rax
	movl	%ecx, (%rax)
	movl	(rip), %eax
	movl	%eax, (%rbp)
	jmp	.LBB1_8
.LBB1_4:
	movsbl	(%rip), %edi
	movb	$0, %al
	callq	sqlite3_free_tablePLT
	cmpl	$0, (%rip)
	je	.LBB1_6
	movq	(%rip), %rax
	movsbl	(%rax), %edi
	movb	$0, %al
	callq	sqlite3_free_tablePLT
.LBB1_6:
	movq	(rip), %rdi
	movslq	(%rbp), %rsi
	shlq	$0, %rsi
	addq	$1, %rsi
	movb	$0, %al
	callq	sqlite3_mallocPLT
	movq	%rax, (%rbp)
	cmpq	$0, (%rbp)
	je	.LBB1_8
	movl	$1, (%rbp)
.LBB1_8:
	movl	(%rbp), %eax
	addq	$48, %rsp
	.p2align	4, 0x90
	movq	(%rbp), %rax
	movl	28(%rax), %eax
	movq	(%rbp), %rcx
	addl	16(%rcx), %eax
	cmpl	$0, %eax
	movq	(%rbp), %rax
	movl	16(%rax), %ecx
	addl	(%rbp), %ecx
	addl	$1, %ecx
	movq	(%rbp), %rax
	movl	%ecx, 16(%rax)
	jmp	.LBB2_14
	movq	(%rbp), %rax
	cmpl	$0, 20(%rax)
	je	.LBB2_10
	movl	$0, (%rbp)
	movq	(%rbp), %rax
	movl	$0, 24(%rax)
.LBB2_4:
	movl	(%rbp), %eax
	cmpl	(%rbp), %eax
	jge	.LBB2_7
	movq	(%rbp), %rax
	movslq	(%rbp), %rcx
	movsbl	(%rax,%rcx), %esi
	leaq	.str, %rdi
	movb	$0, %al
	callq	sqlite3_mprintf@PLT
	movq	%rax, -64(%rbp)
	movl	-56(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -56(%rbp)
	jmp	.LBB2_4
.LBB2_7:
	cmpq	$0, -64(%rbp)
	je	.LBB2_9
	jmp	.LBB2_9
.LBB2_9:
	movq	-64(%rbp), %rdx
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	-48(%rbp), %rsi
	movl	28(%rsi), %ecx
	movl	%ecx, %edi
	addl	$1, %edi
	movl	%edi, 28(%rsi)
	movslq	%ecx, %rcx
	movq	%rdx, (%rax,%rcx,8)
	jmp	.LBB2_13
.LBB2_10:
	movq	-48(%rbp), %rax
	movslq	-20(%rbp), %rcx
	cmpq	%rcx, %rax
	je	.LBB2_12
	movq	-48(%rbp), %rax
	movq	8(%rax), %rdi
	movb	$0, %al
	callq	sqlite3_free_table@PLT
	leaq	.L.str, %rdi
	movb	$0, %al
	callq	sqlite3_mprintf@PLT
	movq	%rax, %rcx
	movq	-48(%rbp), %rax
	movq	%rcx, 8(%rax)
	movq	-48(%rbp), %rax
	movl	$1, 32(%rax)
	movl	$1, -4(%rbp)
	jmp	.LBB2_15
.LBB2_12:
	jmp	.LBB2_13
.LBB2_13:
	movl	$0, -4(%rbp)
	jmp	.LBB2_15
.LBB2_14:
.LBB2_15:
	movl	-4(%rbp), %eax
	addq	$64, %rsp
	popq	%rbp
	.p2align	4, 0x90
sqlite3_complete:
	je	.LBB3_54
	je	.LBB3_25
	je	.LBB3_25
	je	.LBB3_25
	jmp	.LBB3_28
	movl	$42, %eax
	cmpq	%rax, -160
	je	.LBB3_5
	jmp	.LBB3_53
.LBB3_5:
	jmp	.LBB3_6
.LBB3_6:
	movq	-160, %rax
	movsbl	0, %ecx
	xorl	%eax, %eax
	movb	$1, %al
	movl	$42, %ecx
	cmpq	%rcx, -160
	movb	%al, -340
	jne	.LBB3_9
	movq	-160, %rax
	movsbl	10, %eax
	cmpl	$0, %eax
	setne	%al
	movb	%al, -340
.LBB3_9:
	movb	-340, %al
	movb	%al, -330
	movb	-330, %al
	testb	$1, %al
	jne	.LBB3_11
	jmp	.LBB3_14
.LBB3_11:
	movq	-160, %rax
	movsbl	0, %eax
	cmpl	$0, %eax
	jne	.LBB3_13
	movl	$0, -40
	jmp	.LBB3_54
.LBB3_13:
	jmp	.LBB3_6
.LBB3_14:
	jmp	.LBB3_53
	movq	-160, %rax
	movsbl	10, %eax
	cmpl	$45, %eax
	je	.LBB3_17
	movb	$2, -170
	jmp	.LBB3_53
.LBB3_17:
	jmp	.LBB3_18
.LBB3_18:
	movq	-160, %rax
	movsbl	0, %ecx
	xorl	%eax, %eax
	cmpl	$0, %ecx
	movb	%al, -350
	jmp	.LBB3_53
.LBB3_25:
	movl	$0, -40
	jmp	.LBB3_54
	jmp	.LBB3_53
.LBB3_28:
	je	.LBB3_52
	jmp	.LBB3_52
	jmp	.LBB3_31
.LBB3_31:
	movq	-160, %rax
	movslq	-280, %rcx
	movsbl	0, %ecx
	movb	-360, %al
	andb	$1, %cl
	movzbl	%cl, %ecx
	movl	%ecx, -240
	testb	$1, %al
	jne	.LBB3_36
	jmp	.LBB3_51
.LBB3_36:
	movq	-160, %rax
	movsbl	0, %eax
	orl	$32, %eax
	cmpl	$0, -280
	je	.LBB3_40
	movq	-160, %rdi
	leaq	.L.str, %rsi
	movl	$7, %edx
	movb	$0, %al
	callq	sqlite3StrNICmp@PLT
	cmpl	$0, %eax
	je	.LBB3_40
	jmp	.LBB3_48
.LBB3_40:
	cmpl	$0, -280
	je	.LBB3_43
	movq	-160, %rdi
	leaq	.L.str, %rsi
	movl	$4, %edx
	movb	$0, %al
	callq	sqlite3StrNICmp@PLT
	cmpl	$0, %eax
	je	.LBB3_43
	movb	$5, -170
	jmp	.LBB3_47
.LBB3_43:
	cmpl	$0, -280
	je	.LBB3_46
	movq	-160, %rdi
	leaq	.L.str, %rsi
	movl	$9, %edx
	movb	$0, %al
	callq	sqlite3StrNICmp@PLT
	cmpl	$0, %eax
	je	.LBB3_46
	movl	-280, %ecx
	subl	$1, %ecx
	movq	-160, %rax
	movslq	%ecx, %rcx
	addq	%rcx, %rax
	movq	%rax, -160
.LBB3_46:
	jmp	.LBB3_47
.LBB3_47:
	jmp	.LBB3_48
.LBB3_48:
	jmp	.LBB3_49
.LBB3_49:
	jmp	.LBB3_50
.LBB3_50:
	jmp	.LBB3_31
.LBB3_51:
	jmp	.LBB3_52
.LBB3_52:
.LBB3_53:
.LBB3_54:
.L.str:
