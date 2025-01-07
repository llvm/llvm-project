; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp -passes=loop-vectorize -tail-predication=enabled -S < %s | \
; RUN:  FileCheck %s

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

; Check that loop hint predicate.enable loop can overrule the TTI hook. For
; this test case, the TTI hook rejects tail-predication:
;
;   ARMHWLoops: Trip count does not fit into 32bits
;   preferPredicateOverEpilogue: hardware-loop is not profitable.
;
define dso_local void @tail_folding(ptr noalias nocapture %A, ptr noalias nocapture readonly %B, ptr noalias nocapture readonly %C) {
; CHECK-LABEL: tail_folding(
; CHECK:       vector.body:
; CHECK-NOT:   call <4 x i32> @llvm.masked.load.v4i32.p0(
; CHECK-NOT:   call void @llvm.masked.store.v4i32.p0(
; CHECK:       br i1 %{{.*}}, label %{{.*}}, label %vector.body, !llvm.loop [[VEC_LOOP1:![0-9]+]]
;
; CHECK:       br i1 %{{.*}}, label %{{.*}}, label %for.body, !llvm.loop [[SCALAR_LOOP1:![0-9]+]]
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %C, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %add, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 430
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; The same test case but now with predicate.enable = true should get
; tail-folded.
;
define dso_local void @predicate_loop_hint(ptr noalias nocapture %A, ptr noalias nocapture readonly %B, ptr noalias nocapture readonly %C) {
; CHECK-LABEL: predicate_loop_hint(
; CHECK:       vector.body:
; CHECK:         %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:         %[[ELEM0:.*]] = add i64 %index, 0
; CHECK:         %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i64(i64 %[[ELEM0]], i64 430)
; CHECK:         %[[WML1:.*]] = call <4 x i32> @llvm.masked.load.v4i32.p0({{.*}}<4 x i1> %active.lane.mask
; CHECK:         %[[WML2:.*]] = call <4 x i32> @llvm.masked.load.v4i32.p0({{.*}}<4 x i1> %active.lane.mask
; CHECK:         %[[ADD:.*]] = add nsw <4 x i32> %[[WML2]], %[[WML1]]
; CHECK:         call void @llvm.masked.store.v4i32.p0(<4 x i32> %[[ADD]], {{.*}}<4 x i1> %active.lane.mask
; CHECK:         %index.next = add nuw i64 %index, 4
; CHECK:         br i1 %{{.*}}, label %{{.*}}, label %vector.body, !llvm.loop [[VEC_LOOP2:![0-9]+]]
;
; CHECK:         br i1 %{{.*}}, label %{{.*}}, label %for.body, !llvm.loop [[SCALAR_LOOP2:![0-9]+]]
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %C, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %add, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 430
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !6
}

; CHECK:      [[VEC_LOOP1]] = distinct !{[[VEC_LOOP1]], [[MD_IS_VEC:![0-9]+]], [[MD_RT_UNROLL_DIS:![0-9]+]]}
; CHECK-NEXT: [[MD_IS_VEC:![0-9]+]] = !{!"llvm.loop.isvectorized", i32 1}
; CHECK-NEXT: [[MD_RT_UNROLL_DIS]] = !{!"llvm.loop.unroll.runtime.disable"}
; CHECK-NEXT: [[SCALAR_LOOP1]] = distinct !{[[SCALAR_LOOP1]], [[MD_RT_UNROLL_DIS]], [[MD_IS_VEC]]}
; CHECK-NEXT: [[VEC_LOOP2]] = distinct !{[[VEC_LOOP2]], [[MD_IS_VEC]], [[MD_RT_UNROLL_DIS]]}
; CHECK-NEXT: [[SCALAR_LOOP2]] = distinct !{[[SCALAR_LOOP2]], [[MD_RT_UNROLL_DIS]], [[MD_IS_VEC]]}

!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
!8 = !{!"llvm.loop.vectorize.enable", i1 true}
