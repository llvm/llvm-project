; RUN: opt < %s -passes=tsan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare void @escape(ptr)

@sink = global ptr null, align 4

define void @captured0() nounwind uwtable sanitize_thread {
entry:
  %ptr = alloca i32, align 4
  ; escapes due to call
  call void @escape(ptr %ptr)
  store i32 42, ptr %ptr, align 4
  ret void
}
; CHECK-LABEL: define void @captured0
; CHECK: __tsan_write
; CHECK: ret void

define void @captured1() nounwind uwtable sanitize_thread {
entry:
  %ptr = alloca i32, align 4
  ; escapes due to store into global
  store ptr %ptr, ptr @sink, align 8
  store i32 42, ptr %ptr, align 4
  ret void
}
; CHECK-LABEL: define void @captured1
; CHECK: __tsan_write
; CHECK: __tsan_write
; CHECK: ret void

define void @captured2() nounwind uwtable sanitize_thread {
entry:
  %ptr = alloca i32, align 4
  %tmp = alloca ptr, align 8
  ; transitive escape
  store ptr %ptr, ptr %tmp, align 8
  %0 = load ptr, ptr %tmp, align 8
  store ptr %0, ptr @sink, align 8
  store i32 42, ptr %ptr, align 4
  ret void
}
; CHECK-LABEL: define void @captured2
; CHECK: __tsan_write
; CHECK: __tsan_write
; CHECK: ret void

define void @notcaptured0() nounwind uwtable sanitize_thread {
entry:
  %ptr = alloca i32, align 4
  store i32 42, ptr %ptr, align 4
  ; escapes due to call
  call void @escape(ptr %ptr)
  ret void
}
; CHECK-LABEL: define void @notcaptured0
; CHECK: __tsan_write
; CHECK: ret void

define void @notcaptured1() nounwind uwtable sanitize_thread {
entry:
  %ptr = alloca i32, align 4
  store i32 42, ptr %ptr, align 4
  ; escapes due to store into global
  store ptr %ptr, ptr @sink, align 8
  ret void
}
; CHECK-LABEL: define void @notcaptured1
; CHECK: __tsan_write
; CHECK: __tsan_write
; CHECK: ret void

define void @notcaptured2() nounwind uwtable sanitize_thread {
entry:
  %ptr = alloca i32, align 4
  %tmp = alloca ptr, align 8
  store i32 42, ptr %ptr, align 4
  ; transitive escape
  store ptr %ptr, ptr %tmp, align 8
  %0 = load ptr, ptr %tmp, align 8
  store ptr %0, ptr @sink, align 8
  ret void
}
; CHECK-LABEL: define void @notcaptured2
; CHECK: __tsan_write
; CHECK: __tsan_write
; CHECK: ret void

define void @notcaptured3(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  %stkobj = alloca [2 x i32], align 8
  %derived = getelementptr inbounds i32, ptr %stkobj, i64 1
  %ptr = select i1 %cond, ptr %derived, ptr %stkobj
  store i32 42, ptr %ptr, align 4
  ret void
}
; CHECK-LABEL: define void @notcaptured3
; CHECK-NOT: call void @__tsan_write4(ptr %ptr)
; CHECK: ret void

define void @notcaptured4() nounwind uwtable sanitize_thread {
entry:
  %stkobj = alloca [10 x i8], align 1
  br label %loop

exit:
  ret void

loop:
  %count = phi i32 [ 0, %entry ], [ %addone, %loop ]
  %derived = phi ptr [ %stkobj, %entry ], [ %ptraddone, %loop ]
  store i32 %count, ptr %derived, align 4
  %ptraddone = getelementptr inbounds i32, ptr %derived, i64 1
  %addone = add nuw nsw i32 %count, 1
  %eq10 = icmp eq i32 %addone, 10
  br i1 %eq10, label %exit, label %loop
}
; CHECK-LABEL: define void @notcaptured4
; CHECK: ret void
; CHECK-NOT: call void @__tsan_write4(ptr %derived)
