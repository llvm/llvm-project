; RUN: llc < %s | FileCheck %s


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i32 %n) local_unnamed_addr #0 {

; CHECK-LABEL: foo:
; CHECK:       # %bb.0:
; CHECK-NEXT:  	pushq	%rbp
; CHECK-NEXT:  	.cfi_def_cfa_offset 16
; CHECK-NEXT:  	.cfi_offset %rbp, -16
; CHECK-NEXT:  	movq	%rsp, %rbp
; CHECK-NEXT:  	.cfi_def_cfa_register %rbp
; CHECK-NEXT:  	movl	%edi, %eax
; CHECK-NEXT:  	leaq	15(,%rax,4), %rax
; CHECK-NEXT:  	andq	$-16, %rax
; CHECK-NEXT:  	cmpq	$4096, %rax # imm = 0x1000
; CHECK-NEXT:  	jl	.LBB0_3
; CHECK-NEXT:  .LBB0_2: # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:  	subq	$4096, %rax # imm = 0x1000
; CHECK-NEXT:  	subq	$4096, %rsp # imm = 0x1000
; CHECK-NEXT:  	movq	$0, (%rsp)
; CHECK-NEXT:  	cmpq	$4096, %rax # imm = 0x1000
; CHECK-NEXT:  	jge	.LBB0_2
; CHECK-NEXT:  .LBB0_3:
; CHECK-NEXT:  	subq	%rax, %rsp
; CHECK-NEXT:  	movq	%rsp, %rax
; CHECK-NEXT:  	movl	$1, 4792(%rax)
; CHECK-NEXT:  	movl	(%rax), %eax
; CHECK-NEXT:  	movq	%rbp, %rsp
; CHECK-NEXT:  	popq	%rbp
; CHECK-NEXT:  .cfi_def_cfa %rsp, 8
; CHECK-NEXT:   retq

  %a = alloca i32, i32 %n, align 16
  %b = getelementptr inbounds i32, i32* %a, i64 1198
  store volatile i32 1, i32* %b
  %c = load volatile i32, i32* %a
  ret i32 %c
}

attributes #0 =  {"probe-stack"="inline-asm"}
