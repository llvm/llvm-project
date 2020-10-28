; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp -loop-vectorize -tail-predication=enabled -S < %s | \
; RUN:  FileCheck %s

; Check that loop hint predicate.enable loop can overrule the TTI hook. For
; this test case, the TTI hook rejects tail-predication:
;
;   ARMHWLoops: Trip count does not fit into 32bits
;   preferPredicateOverEpilogue: hardware-loop is not profitable.
;
define dso_local void @tail_folding(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) {
; CHECK-LABEL: tail_folding(
; CHECK:       vector.body:
; CHECK-NOT:   call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(
; CHECK-NOT:   call void @llvm.masked.store.v4i32.p0v4i32(
; CHECK:       br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %add, i32* %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 430
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; The same test case but now with predicate.enable = true should get
; tail-folded.
;
define dso_local void @predicate_loop_hint(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) {
; CHECK-LABEL: predicate_loop_hint(
; CHECK:       vector.body:
; CHECK:         %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:         %[[ELEM0:.*]] = add i64 %index, 0
; CHECK:         %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i64(i64 %[[ELEM0]], i64 429)
; CHECK:         %[[WML1:.*]] = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}<4 x i1> %active.lane.mask
; CHECK:         %[[WML2:.*]] = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}<4 x i1> %active.lane.mask
; CHECK:         %[[ADD:.*]] = add nsw <4 x i32> %[[WML2]], %[[WML1]]
; CHECK:         call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %[[ADD]], {{.*}}<4 x i1> %active.lane.mask
; CHECK:         %index.next = add i64 %index, 4
; CHECK:         br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %add, i32* %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 430
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !6
}

; CHECK:      !0 = distinct !{!0, !1}
; CHECK-NEXT: !1 = !{!"llvm.loop.isvectorized", i32 1}
; CHECK-NEXT: !2 = distinct !{!2, !3, !1}
; CHECK-NEXT: !3 = !{!"llvm.loop.unroll.runtime.disable"}
; CHECK-NEXT: !4 = distinct !{!4, !1}
; CHECK-NEXT: !5 = distinct !{!5, !3, !1}

!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
!8 = !{!"llvm.loop.vectorize.enable", i1 true}
