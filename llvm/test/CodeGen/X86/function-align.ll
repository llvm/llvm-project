; RUN: llc -function-sections < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: .section .text.f1
; CHECK-NOT: .p2align
; CHECK: f1:
define void @f1() align 1 {
  ret void
}

; CHECK: .section .text.f2
; CHECK-NEXT: .globl f2
; CHECK-NEXT: .p2align 1
define void @f2() align 2 {
  ret void
}
