; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i64 %i) local_unnamed_addr #0 {
; CHECK-LABEL: foo:
; CHECK:       # %bb.0:
; CHECK-NEXT:  subq	$4096, %rsp # imm = 0x1000
; CHECK-NEXT:  movq	$0, (%rsp)
; CHECK-NEXT:  subq	$3784, %rsp # imm = 0xEC8
; CHECK-NEXT:  .cfi_def_cfa_offset 7888
; CHECK-NEXT:  movl	$1, -128(%rsp,%rdi,4)
; CHECK-NEXT:  movl	-128(%rsp), %eax
; CHECK-NEXT:  addq	$7880, %rsp # imm = 0x1EC8
; CHECK-NEXT:  .cfi_def_cfa_offset 8
; CHECK-NEXT:  retq

  %a = alloca i32, i32 2000, align 16
  %b = getelementptr inbounds i32, i32* %a, i64 %i
  store volatile i32 1, i32* %b
  %c = load volatile i32, i32* %a
  ret i32 %c
}

attributes #0 =  {"probe-stack"="inline-asm"}

