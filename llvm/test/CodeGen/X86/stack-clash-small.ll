; RUN: llc < %s | FileCheck %s


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo() local_unnamed_addr #0 {
; CHECK-LABEL: foo:
; CHECK:       # %bb.0:
; CHECK-NEXT:  subq	$280, %rsp # imm = 0x118
; CHECK-NEXT:  .cfi_def_cfa_offset 288
; CHECK-NEXT:  movl	$1, 264(%rsp)
; CHECK-NEXT:  movl	-128(%rsp), %eax
; CHECK-NEXT:  addq	$280, %rsp # imm = 0x118
; CHECK-NEXT:  .cfi_def_cfa_offset 8
; CHECK-NEXT:  retq

  %a = alloca i32, i64 100, align 16
  %b = getelementptr inbounds i32, i32* %a, i64 98
  store volatile i32 1, i32* %b
  %c = load volatile i32, i32* %a
  ret i32 %c
}

attributes #0 =  {"probe-stack"="inline-asm"}
