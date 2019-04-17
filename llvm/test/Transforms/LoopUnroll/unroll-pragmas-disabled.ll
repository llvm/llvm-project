; RUN: opt < %s -loop-unroll -S | FileCheck %s
;
; Verify that the unrolling pass removes existing unroll count metadata
; and adds a disable unrolling node after unrolling is complete.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; #pragma clang loop  vectorize(enable) unroll_count(4) vectorize_width(8)
;
; Unroll count metadata should be replaced with unroll(disable).  Vectorize
; metadata should be untouched.
;
; CHECK-LABEL: @unroll_count_4(
; CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_1:.*]]
define void @unroll_count_4(i32* nocapture %a) {
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
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:                                          ; preds = %for.body
  ret void
}
!1 = !{!1, !2, !3, !4}
!2 = !{!"llvm.loop.vectorize.enable", i1 true}
!3 = !{!"llvm.loop.unroll.count", i32 4}
!4 = !{!"llvm.loop.vectorize.width", i32 8}

; #pragma clang loop unroll(full)
;
; An unroll disable metadata node is only added for the unroll count case.
; In this case, the loop has a full unroll metadata but can't be fully unrolled
; because the trip count is dynamic.  The full unroll metadata should remain
; after unrolling.
;
; CHECK-LABEL: @unroll_full(
; CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_2:.*]]
define void @unroll_full(i32* nocapture %a, i32 %b) {
entry:
  %cmp3 = icmp sgt i32 %b, 0
  br i1 %cmp3, label %for.body, label %for.end, !llvm.loop !5

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %b
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !5

for.end:                                          ; preds = %for.body, %entry
  ret void
}
!5 = !{!5, !6}
!6 = !{!"llvm.loop.unroll.full"}

; #pragma clang loop unroll(disable)
;
; Unroll metadata should not change.
;
; CHECK-LABEL: @unroll_disable(
; CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_3:.*]]
define void @unroll_disable(i32* nocapture %a) {
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
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !7

for.end:                                          ; preds = %for.body
  ret void
}
!7 = !{!7, !8}
!8 = !{!"llvm.loop.unroll.disable"}

; This function contains two loops which share the same llvm.loop metadata node
; with an llvm.loop.unroll.count 2 hint.  Both loops should be unrolled.  This
; verifies that adding disable metadata to a loop after unrolling doesn't affect
; other loops which previously shared the same llvm.loop metadata.
;
; CHECK-LABEL: @shared_metadata(
; CHECK: store i32
; CHECK: store i32
; CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_4:.*]]
; CHECK: store i32
; CHECK: store i32
; CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_5:.*]]
define void @shared_metadata(i32* nocapture %List) #0 {
entry:
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx = getelementptr inbounds i32, i32* %List, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add4 = add nsw i32 %0, 10
  store i32 %add4, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 4
  br i1 %exitcond, label %for.body3.1.preheader, label %for.body3, !llvm.loop !9

for.body3.1.preheader:                            ; preds = %for.body3
  br label %for.body3.1

for.body3.1:                                      ; preds = %for.body3.1.preheader, %for.body3.1
  %indvars.iv.1 = phi i64 [ %1, %for.body3.1 ], [ 0, %for.body3.1.preheader ]
  %1 = add nsw i64 %indvars.iv.1, 1
  %arrayidx.1 = getelementptr inbounds i32, i32* %List, i64 %1
  %2 = load i32, i32* %arrayidx.1, align 4
  %add4.1 = add nsw i32 %2, 10
  store i32 %add4.1, i32* %arrayidx.1, align 4
  %exitcond.1 = icmp eq i64 %1, 4
  br i1 %exitcond.1, label %for.inc5.1, label %for.body3.1, !llvm.loop !9

for.inc5.1:                                       ; preds = %for.body3.1
  ret void
}
!9 = !{!9, !10}
!10 = !{!"llvm.loop.unroll.count", i32 2}


; CHECK: ![[LOOP_1]] = distinct !{![[LOOP_1]], ![[VEC_ENABLE:.*]], ![[WIDTH_8:.*]], ![[UNROLL_DISABLE:.*]]}
; CHECK: ![[VEC_ENABLE]] = !{!"llvm.loop.vectorize.enable", i1 true}
; CHECK: ![[WIDTH_8]] = !{!"llvm.loop.vectorize.width", i32 8}
; CHECK: ![[UNROLL_DISABLE]] = !{!"llvm.loop.unroll.disable"}
; CHECK: ![[LOOP_2]] = distinct !{![[LOOP_2]], ![[UNROLL_FULL:.*]]}
; CHECK: ![[UNROLL_FULL]] = !{!"llvm.loop.unroll.full"}
; CHECK: ![[LOOP_3]] = distinct !{![[LOOP_3]], ![[UNROLL_DISABLE:.*]]}
; CHECK: ![[LOOP_4]] = distinct !{![[LOOP_4]], ![[UNROLL_DISABLE:.*]]}
; CHECK: ![[LOOP_5]] = distinct !{![[LOOP_5]], ![[UNROLL_DISABLE:.*]]}
