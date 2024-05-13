; RUN: opt -S -passes=instnamer < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f_0(i32) {
; CHECK-LABEL: @f_0(
; CHECK: bb:
; CHECK-NEXT:   %i = add i32 %arg, 2
; CHECK-NEXT:   br label %bb1
; CHECK: bb1:
; CHECK-NEXT:   ret i32 %i

  %2 = add i32 %0, 2
  br label %3

; <label>:3:
  ret i32 %2
}
