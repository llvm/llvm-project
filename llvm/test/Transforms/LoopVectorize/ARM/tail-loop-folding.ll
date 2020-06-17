; RUN: opt < %s -loop-vectorize -disable-mve-tail-predication=false -S | \
; RUN:  FileCheck %s -check-prefixes=COMMON,CHECK

; RUN: opt < %s -loop-vectorize -disable-mve-tail-predication=false -prefer-predicate-over-epilog -S | \
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
; COMMON:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; COMMON:   %[[ELEM0:.*]] = add i64 %index, 0
; COMMON:   %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i64(i64 %[[ELEM0]], i64 429)
; COMMON:   %[[WML1:.*]] = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}<4 x i1> %active.lane.mask
; COMMON:   %[[WML2:.*]] = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}<4 x i1> %active.lane.mask
; COMMON:   %[[ADD:.*]] = add nsw <4 x i32> %[[WML2]], %[[WML1]]
; COMMON:   call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %[[ADD]], {{.*}}<4 x i1> %active.lane.mask
; COMMON:   %index.next = add i64 %index, 4
; COMMON:   br i1 %{{.*}}, label %{{.*}}, label %vector.body
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
; PREDFLAG:  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; PREDFLAG:  %[[ELEM0:.*]] = add i64 %index, 0
; PREDFLAG:  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i64(i64 %[[ELEM0]], i64 429)
; PREDFLAG:  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}, <4 x i1> %active.lane.mask
; PREDFLAG:  %wide.masked.load1 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}, <4 x i1> %active.lane.mask
; PREDFLAG:  %{{.*}} = add nsw <4 x i32> %wide.masked.load1, %wide.masked.load
; PREDFLAG:  call void @llvm.masked.store.v4i32.p0v4i32({{.*}}, <4 x i1> %active.lane.mask
; PREDFLAG:  %index.next = add i64 %index, 4
; PREDFLAG:  %[[CMP:.*]] = icmp eq i64 %index.next, 432
; PREDFLAG:  br i1 %[[CMP]], label %middle.block, label %vector.body, !llvm.loop !6
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

define dso_local void @interleave4(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i32 %N) local_unnamed_addr #0 {
; PREDFLAG-LABEL: interleave4(
; PREDFLAG:  %[[ADD1:.*]] = add i32 %index, 0
; PREDFLAG:  %[[ADD2:.*]] = add i32 %index, 4
; PREDFLAG:  %[[ADD3:.*]] = add i32 %index, 8
; PREDFLAG:  %[[ADD4:.*]] = add i32 %index, 12
; PREDFLAG:  %[[BTC:.*]] = extractelement <4 x i32> %broadcast.splat, i32 0
; PREDFLAG:  %[[ALM1:active.lane.mask.*]] = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %[[ADD1]], i32 %[[BTC]])
; PREDFLAG:  %[[ALM2:active.lane.mask.*]] = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %[[ADD2]], i32 %[[BTC]])
; PREDFLAG:  %[[ALM3:active.lane.mask.*]] = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %[[ADD3]], i32 %[[BTC]])
; PREDFLAG:  %[[ALM4:active.lane.mask.*]] = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %[[ADD4]], i32 %[[BTC]])
;
; PREDFLAG:  call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}, <4 x i1> %[[ALM1]],{{.*}}
; PREDFLAG:  call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}, <4 x i1> %[[ALM2]],{{.*}}
; PREDFLAG:  call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}, <4 x i1> %[[ALM3]],{{.*}}
; PREDFLAG:  call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}, <4 x i1> %[[ALM4]],{{.*}}
; PREDFLAG:  call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}, <4 x i1> %[[ALM1]],{{.*}}
; PREDFLAG:  call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}, <4 x i1> %[[ALM2]],{{.*}}
; PREDFLAG:  call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}, <4 x i1> %[[ALM3]],{{.*}}
; PREDFLAG:  call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}, <4 x i1> %[[ALM4]],{{.*}}
;
; PREDFLAG:  call void @llvm.masked.store.v4i32.p0v4i32({{.*}}, <4 x i1> %[[ALM1]])
; PREDFLAG:  call void @llvm.masked.store.v4i32.p0v4i32({{.*}}, <4 x i1> %[[ALM2]])
; PREDFLAG:  call void @llvm.masked.store.v4i32.p0v4i32({{.*}}, <4 x i1> %[[ALM3]])
; PREDFLAG:  call void @llvm.masked.store.v4i32.p0v4i32({{.*}}, <4 x i1> %[[ALM4]])
;
entry:
  %cmp8 = icmp sgt i32 %N, 0
  br i1 %cmp8, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !14
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

!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.interleave.count", i32 4}
