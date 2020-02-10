; RUN: llc < %s | FileCheck %s


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo() local_unnamed_addr #0 {

; CHECK-LABEL: foo:
; CHECK:        # %bb.0:
; CHECK-NEXT:	movq	%rsp, %r11
; CHECK-NEXT:	subq	$69632, %r11 # imm = 0x11000
; CHECK-NEXT:   .LBB0_1:
; CHECK-NEXT:	subq	$4096, %rsp # imm = 0x1000
; CHECK-NEXT:	movq	$0, (%rsp)
; CHECK-NEXT:	cmpq	%r11, %rsp
; CHECK-NEXT:	jne	.LBB0_1
; CHECK-NEXT:# %bb.2:
; CHECK-NEXT:	subq	$2248, %rsp # imm = 0x8C8
; CHECK-NEXT:	.cfi_def_cfa_offset 71888
; CHECK-NEXT:	movl	$1, 264(%rsp)
; CHECK-NEXT:	movl	$1, 28664(%rsp)
; CHECK-NEXT:	movl	-128(%rsp), %eax
; CHECK-NEXT:	addq	$71880, %rsp # imm = 0x118C8
; CHECK-NEXT:	.cfi_def_cfa_offset 8
; CHECK-NEXT:	retq


  %a = alloca i32, i64 18000, align 16
  %b0 = getelementptr inbounds i32, i32* %a, i64 98
  %b1 = getelementptr inbounds i32, i32* %a, i64 7198
  store volatile i32 1, i32* %b0
  store volatile i32 1, i32* %b1
  %c = load volatile i32, i32* %a
  ret i32 %c
}

attributes #0 =  {"probe-stack"="inline-asm"}
