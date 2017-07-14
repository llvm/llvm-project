; RUN: llc -o - -mtriple=x86_64-linux-gnu %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; We should be able to bypass the load values to their corresponding
; stores here.

; CHECK-LABEL: foo
; CHECK-DAG: movl	%esi, -8(%rsp)
; CHECK-DAG: movl	%ecx, -16(%rsp)
; CHECK-DAG: movl	%edi, -4(%rsp)
; CHECK-DAG: movl	%edx, -12(%rsp)
; CHECK: leal
; CHECK: addl
; CHECK: addl
; CHECK: retq

define i32 @foo(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %a0 = alloca i32
  %a1 = alloca i32
  %a2 = alloca i32
  %a3 = alloca i32
  store i32 %b, i32* %a1
  store i32 %d, i32* %a3
  store i32 %a, i32* %a0
  store i32 %c, i32* %a2
  %l0 = load i32, i32* %a0
  %l1 = load i32, i32* %a1
  %l2 = load i32, i32* %a2
  %l3 = load i32, i32* %a3
  %add0 = add nsw i32 %l0, %l1
  %add1 = add nsw i32 %add0, %l2
  %add2 = add nsw i32 %add1, %l3
  ret i32 %add2
}
