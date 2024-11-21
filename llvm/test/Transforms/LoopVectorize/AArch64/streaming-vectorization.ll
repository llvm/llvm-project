; REQUIRES: asserts
; RUN: opt -S -passes=loop-vectorize -debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOVEC
; RUN: opt -S -passes=loop-vectorize -debug-only=loop-vectorize -enable-scalable-autovec-in-streaming-mode < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,VEC

target triple = "aarch64-unknown-linux-gnu"

define void @normal_function(ptr %a, ptr %b, ptr %c) #0 {
; CHECK: LV: Checking a loop in 'normal_function'
; CHECK: LV: Scalable vectorization is available
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %iv
  %1 = load i8, ptr %arrayidx2, align 4
  %zext = zext i8 %1 to i32
  %add = add nsw i32 %zext, %0
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @streaming_function(ptr %a, ptr %b, ptr %c) #0 "aarch64_pstate_sm_enabled" {
; CHECK: LV: Checking a loop in 'streaming_function'
; VEC: LV: Scalable vectorization is available
; NOVEC: LV: Scalable vectorization is explicitly disabled
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %iv
  %1 = load i8, ptr %arrayidx2, align 4
  %zext = zext i8 %1 to i32
  %add = add nsw i32 %zext, %0
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

attributes #0 = { vscale_range(1, 16) "target-features"="+sve,+sme" }
