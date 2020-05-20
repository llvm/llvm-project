; RUN: opt < %s -tsan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @IncrementMe(i32* nocapture %ptr) nounwind uwtable sanitize_thread {
entry:
  %0 = load i32, i32* %ptr, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %ptr, align 4
  ret void
}
; CHECK: define void @IncrementMe
; CHECK-NOT: __tsan_read
; CHECK: __tsan_write
; CHECK: ret void

define void @IncrementMeWithCallInBetween(i32* nocapture %ptr) nounwind uwtable sanitize_thread {
entry:
  %0 = load i32, i32* %ptr, align 4
  %inc = add nsw i32 %0, 1
  call void @foo()
  store i32 %inc, i32* %ptr, align 4
  ret void
}

; CHECK: define void @IncrementMeWithCallInBetween
; CHECK: __tsan_read
; CHECK: __tsan_write
; CHECK: ret void

declare void @foo()

