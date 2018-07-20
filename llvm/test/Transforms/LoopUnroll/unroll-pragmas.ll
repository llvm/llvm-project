; RUN: opt < %s -loop-unroll -pragma-unroll-threshold=1024 -S | FileCheck -check-prefixes=CHECK,REM %s
; RUN: opt < %s -loop-unroll -loop-unroll -pragma-unroll-threshold=1024 -S | FileCheck -check-prefixes=CHECK,REM %s
; RUN: opt < %s -loop-unroll -unroll-allow-remainder=0 -pragma-unroll-threshold=1024 -S | FileCheck -check-prefixes=CHECK,NOREM %s
;
; Run loop unrolling twice to verify that loop unrolling metadata is properly
; removed and further unrolling is disabled after the pass is run once.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; loop4 contains a small loop which should be completely unrolled by
; the default unrolling heuristics.  It serves as a control for the
; unroll(disable) pragma test loop4_with_disable.
;
; CHECK-LABEL: @loop4(
; CHECK-NOT: br i1
define void @loop4(i32* nocapture %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 4
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; #pragma clang loop unroll(disable)
;
; CHECK-LABEL: @loop4_with_disable(
; CHECK: store i32
; CHECK-NOT: store i32
; CHECK: br i1
define void @loop4_with_disable(i32* nocapture %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 4
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:                                          ; preds = %for.body
  ret void
}
!1 = !{!1, !2}
!2 = !{!"llvm.loop.unroll.disable"}

; loop64 has a high enough count that it should *not* be unrolled by
; the default unrolling heuristic.  It serves as the control for the
; unroll(full) pragma test loop64_with_.* tests below.
;
; CHECK-LABEL: @loop64(
; CHECK: store i32
; CHECK-NOT: store i32
; CHECK: br i1
define void @loop64(i32* nocapture %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; #pragma clang loop unroll(full)
; Loop should be fully unrolled.
;
; CHECK-LABEL: @loop64_with_full(
; CHECK-NOT: br i1
define void @loop64_with_full(i32* nocapture %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !3

for.end:                                          ; preds = %for.body
  ret void
}
!3 = !{!3, !4}
!4 = !{!"llvm.loop.unroll.full"}

; #pragma clang loop unroll_count(4)
; Loop should be unrolled 4 times.
;
; CHECK-LABEL: @loop64_with_count4(
; CHECK: store i32
; CHECK: store i32
; CHECK: store i32
; CHECK: store i32
; CHECK-NOT: store i32
; CHECK: br i1
define void @loop64_with_count4(i32* nocapture %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !5

for.end:                                          ; preds = %for.body
  ret void
}
!5 = !{!5, !6}
!6 = !{!"llvm.loop.unroll.count", i32 4}

; #pragma clang loop unroll(full)
; Full unrolling is requested, but loop has a runtime trip count so
; no unrolling should occur.
;
; CHECK-LABEL: @runtime_loop_with_full(
; CHECK: store i32
; CHECK-NOT: store i32
define void @runtime_loop_with_full(i32* nocapture %a, i32 %b) {
entry:
  %cmp3 = icmp sgt i32 %b, 0
  br i1 %cmp3, label %for.body, label %for.end, !llvm.loop !8

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %b
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !8

for.end:                                          ; preds = %for.body, %entry
  ret void
}
!8 = !{!8, !4}

; #pragma clang loop unroll_count(4)
; Loop has a runtime trip count.  Runtime unrolling should occur and loop
; should be duplicated (original and 4x unrolled) if remainder is allowed,
; otherwise loop should not be unrolled.
;
; CHECK-LABEL: @runtime_loop_with_count4(
; CHECK: for.body
; CHECK: store
; REM: store
; REM: store
; REM: store
; CHECK-NOT: store
; CHECK: br i1
; REM: for.body.epil:
; REM: store
; NOREM-NOT: for.body.epil:
; NOREM-NOT: store
; CHECK-NOT: store
; REM: br i1
; NOREM-NOT: br i1
define void @runtime_loop_with_count4(i32* nocapture %a, i32 %b) {
entry:
  %cmp3 = icmp sgt i32 %b, 0
  br i1 %cmp3, label %for.body, label %for.end, !llvm.loop !9

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %b
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !9

for.end:                                          ; preds = %for.body, %entry
  ret void
}
!9 = !{!9, !6}

; #pragma clang loop unroll_count(1)
; Loop should not be unrolled
;
; CHECK-LABEL: @unroll_1(
; CHECK: store i32
; CHECK-NOT: store i32
; CHECK: br i1
define void @unroll_1(i32* nocapture %a, i32 %b) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 4
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !10

for.end:                                          ; preds = %for.body
  ret void
}
!10 = !{!10, !11}
!11 = !{!"llvm.loop.unroll.count", i32 1}

; #pragma clang loop unroll(full)
; Loop has very high loop count (1 million) and full unrolling was requested.
; Loop should unrolled up to the pragma threshold, but not completely.
;
; CHECK-LABEL: @unroll_1M(
; CHECK: store i32
; CHECK: store i32
; CHECK: br i1
define void @unroll_1M(i32* nocapture %a, i32 %b) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !12

for.end:                                          ; preds = %for.body
  ret void
}
!12 = !{!12, !4}

; #pragma clang loop unroll(enable)
; Loop should be fully unrolled.
;
; CHECK-LABEL: @loop64_with_enable(
; CHECK-NOT: br i1
define void @loop64_with_enable(i32* nocapture %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !13

for.end:                                          ; preds = %for.body
  ret void
}
!13 = !{!13, !14}
!14 = !{!"llvm.loop.unroll.enable"}

; #pragma clang loop unroll(enable)
; Loop has a runtime trip count and should be runtime unrolled and duplicated
; (original and 8x) if remainder is allowed, otherwise it should not be
; unrolled.
;
; CHECK-LABEL: @runtime_loop_with_enable(
; CHECK: for.body:
; CHECK: store i32
; REM: store i32
; REM: store i32
; REM: store i32
; REM: store i32
; REM: store i32
; REM: store i32
; REM: store i32
; CHECK-NOT: store i32
; CHECK: br i1
; REM: for.body.epil:
; NOREM-NOT: for.body.epil:
; REM: store
; CHECK-NOT: store
; REM: br i1
; NOREM-NOT: br i1
define void @runtime_loop_with_enable(i32* nocapture %a, i32 %b) {
entry:
  %cmp3 = icmp sgt i32 %b, 0
  br i1 %cmp3, label %for.body, label %for.end, !llvm.loop !8

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %b
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !15

for.end:                                          ; preds = %for.body, %entry
  ret void
}
!15 = !{!15, !14}

; #pragma clang loop unroll_count(3)
; Loop has a runtime trip count.  Runtime unrolling should occur and loop
; should be duplicated (original and 3x unrolled) if remainder is allowed,
; otherwise it should not be unrolled.
;
; CHECK-LABEL: @runtime_loop_with_count3(
; CHECK: for.body
; CHECK: store
; REM: store
; REM: store
; CHECK-NOT: store
; CHECK: br i1
; REM: for.body.epil:
; REM: store
; NOREM-NOT: for.body.epil:
; NOREM-NOT: store
; CHECK-NOT: store
; REM: br i1
define void @runtime_loop_with_count3(i32* nocapture %a, i32 %b) {
entry:
  %cmp3 = icmp sgt i32 %b, 0
  br i1 %cmp3, label %for.body, label %for.end, !llvm.loop !16

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %b
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !16

for.end:                                          ; preds = %for.body, %entry
  ret void
}
!16 = !{!16, !17}
!17 = !{!"llvm.loop.unroll.count", i32 3}
