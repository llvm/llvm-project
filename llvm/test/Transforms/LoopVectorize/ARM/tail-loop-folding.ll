; RUN: opt < %s -loop-vectorize -S | \
; RUN:  FileCheck %s -check-prefixes=COMMON,CHECK

; RUN: opt < %s -loop-vectorize -prefer-predicate-over-epilog -S | \
; RUN:   FileCheck -check-prefixes=COMMON,PREDFLAG %s

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-unknown-eabihf"

define dso_local void @tail_folding(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL: tail_folding(
; CHECK: vector.body:
;
; This needs implementation of TTI::preferPredicateOverEpilogue,
; then this will be tail-folded too:
;
; CHECK-NOT:  call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(
; CHECK-NOT:  call void @llvm.masked.store.v4i32.p0v4i32(
; CHECK:      br i1 %{{.*}}, label %{{.*}}, label %vector.body
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


define dso_local void @tail_folding_enabled(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) local_unnamed_addr #0 {
; COMMON-LABEL: tail_folding_enabled(
; COMMON: vector.body:
; COMMON:   %[[WML1:.*]] = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(
; COMMON:   %[[WML2:.*]] = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(
; COMMON:   %[[ADD:.*]] = add nsw <4 x i32> %[[WML2]], %[[WML1]]
; COMMON:   call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %[[ADD]]
; COMMON:   br i1 %12, label %{{.*}}, label %vector.body
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

define dso_local void @tail_folding_disabled(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) local_unnamed_addr #0 {
; CHECK-LABEL: tail_folding_disabled(
; CHECK:      vector.body:
; CHECK-NOT:  @llvm.masked.load.v8i32.p0v8i32(
; CHECK-NOT:  @llvm.masked.store.v8i32.p0v8i32(
; CHECK:      br i1 %{{.*}}, label {{.*}}, label %vector.body

; PREDFLAG-LABEL: tail_folding_disabled(
; PREDFLAG:  vector.body:
; PREDFLAG:  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(
; PREDFLAG:  %wide.masked.load1 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(
; PREDFLAG:  %{{.*}} = add nsw <4 x i32> %wide.masked.load1, %wide.masked.load
; PREDFLAG:  call void @llvm.masked.store.v4i32.p0v4i32(
; PREDFLAG:  %index.next = add i64 %index, 4
; PREDFLAG:  %12 = icmp eq i64 %index.next, 432
; PREDFLAG:  br i1 %{{.*}}, label %middle.block, label %vector.body, !llvm.loop !6
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
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !10
}

; CHECK:      !0 = distinct !{!0, !1}
; CHECK-NEXT: !1 = !{!"llvm.loop.isvectorized", i32 1}
; CHECK-NEXT: !2 = distinct !{!2, !3, !1}
; CHECK-NEXT: !3 = !{!"llvm.loop.unroll.runtime.disable"}
; CHECK-NEXT: !4 = distinct !{!4, !1}
; CHECK-NEXT: !5 = distinct !{!5, !3, !1}
; CHECK-NEXT: !6 = distinct !{!6, !1}
attributes #0 = { nofree norecurse nounwind "target-features"="+armv8.1-m.main,+mve.fp" }

!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
!8 = !{!"llvm.loop.vectorize.enable", i1 true}

!10 = distinct !{!10, !11, !12}
!11 = !{!"llvm.loop.vectorize.predicate.enable", i1 false}
!12 = !{!"llvm.loop.vectorize.enable", i1 true}
