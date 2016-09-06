; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=g5 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i64 @foo(i64 %a) nounwind {
entry:
  call void asm sideeffect "", "~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12}"() nounwind
  br label %return

; CHECK: @foo
; Because r0 is allocatable, we can use it to hold r3 without spilling.
; CHECK: mr 0, 3
; CHECK: mr 3, 0

return:                                           ; preds = %entry
  ret i64 %a
}

