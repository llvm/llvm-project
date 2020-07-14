; RUN: opt < %s -loop-vectorize -tail-predication=enabled -S | \
; RUN:  FileCheck %s -check-prefixes=COMMON,CHECK

; RUN: opt < %s -loop-vectorize -tail-predication=enabled -prefer-predicate-over-epilog -S | \
; RUN:   FileCheck -check-prefixes=COMMON,PREDFLAG %s

; RUN: opt < %s -loop-vectorize -tail-predication=enabled-no-reductions -S | \
; RUN:  FileCheck %s -check-prefixes=COMMON,NORED

; RUN: opt < %s -loop-vectorize -tail-predication=force-enabled-no-reductions -S | \
; RUN:  FileCheck %s -check-prefixes=COMMON,NORED


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

define dso_local i32 @i32_add_reduction(i32* noalias nocapture readonly %B, i32 %N) local_unnamed_addr #0 {
; COMMON-LABEL: i32_add_reduction(
; COMMON:       entry:
; CHECK:        @llvm.get.active.lane.mask
; NORED-NOT:    @llvm.get.active.lane.mask
; COMMON:       }
entry:
  %cmp6 = icmp sgt i32 %N, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.cleanup.loopexit:
  %add.lcssa = phi i32 [ %add, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:
  %S.0.lcssa = phi i32 [ 1, %entry ], [ %add.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %S.0.lcssa

for.body:
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %S.07 = phi i32 [ %add, %for.body ], [ 1, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %S.07
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}


; Don't tail-fold float reductions.
;
define dso_local void @f32_reduction(float* nocapture readonly %Input, i32 %N, float* nocapture %Output) local_unnamed_addr #0 {
; CHECK-LABEL: f32_reduction(
; CHECK:       vector.body:
; CHECK-NOT:   @llvm.masked.load
; CHECK-NOT:   @llvm.masked.store
; CHECK:       br i1 %{{.*}}, label {{.*}}, label %vector.body
entry:
  %cmp6 = icmp eq i32 %N, 0
  br i1 %cmp6, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %blkCnt.09 = phi i32 [ %dec, %while.body ], [ %N, %while.body.preheader ]
  %sum.08 = phi float [ %add, %while.body ], [ 0.000000e+00, %while.body.preheader ]
  %Input.addr.07 = phi float* [ %incdec.ptr, %while.body ], [ %Input, %while.body.preheader ]
  %incdec.ptr = getelementptr inbounds float, float* %Input.addr.07, i32 1
  %0 = load float, float* %Input.addr.07, align 4
  %add = fadd fast float %0, %sum.08
  %dec = add i32 %blkCnt.09, -1
  %cmp = icmp eq i32 %dec, 0
  br i1 %cmp, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  %add.lcssa = phi float [ %add, %while.body ]
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add.lcssa, %while.end.loopexit ]
  %conv = uitofp i32 %N to float
  %div = fdiv fast float %sum.0.lcssa, %conv
  store float %div, float* %Output, align 4
  ret void
}

; Don't tail-fold float reductions.
;
define dso_local void @mixed_f32_i32_reduction(float* nocapture readonly %fInput, i32* nocapture readonly %iInput, i32 %N, float* nocapture %fOutput, i32* nocapture %iOutput) local_unnamed_addr #0 {
; CHECK-LABEL: mixed_f32_i32_reduction(
; CHECK:       vector.body:
; CHECK-NOT:   @llvm.masked.load
; CHECK-NOT:   @llvm.masked.store
; CHECK:       br i1 %{{.*}}, label {{.*}}, label %vector.body
entry:
  %cmp15 = icmp eq i32 %N, 0
  br i1 %cmp15, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %blkCnt.020 = phi i32 [ %dec, %while.body ], [ %N, %while.body.preheader ]
  %isum.019 = phi i32 [ %add2, %while.body ], [ 0, %while.body.preheader ]
  %fsum.018 = phi float [ %add, %while.body ], [ 0.000000e+00, %while.body.preheader ]
  %fInput.addr.017 = phi float* [ %incdec.ptr, %while.body ], [ %fInput, %while.body.preheader ]
  %iInput.addr.016 = phi i32* [ %incdec.ptr1, %while.body ], [ %iInput, %while.body.preheader ]
  %incdec.ptr = getelementptr inbounds float, float* %fInput.addr.017, i32 1
  %incdec.ptr1 = getelementptr inbounds i32, i32* %iInput.addr.016, i32 1
  %0 = load i32, i32* %iInput.addr.016, align 4
  %add2 = add nsw i32 %0, %isum.019
  %1 = load float, float* %fInput.addr.017, align 4
  %add = fadd fast float %1, %fsum.018
  %dec = add i32 %blkCnt.020, -1
  %cmp = icmp eq i32 %dec, 0
  br i1 %cmp, label %while.end.loopexit, label %while.body

while.end.loopexit:
  %add.lcssa = phi float [ %add, %while.body ]
  %add2.lcssa = phi i32 [ %add2, %while.body ]
  %phitmp = sitofp i32 %add2.lcssa to float
  br label %while.end

while.end:
  %fsum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add.lcssa, %while.end.loopexit ]
  %isum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %phitmp, %while.end.loopexit ]
  %conv = uitofp i32 %N to float
  %div = fdiv fast float %fsum.0.lcssa, %conv
  store float %div, float* %fOutput, align 4
  %div5 = fdiv fast float %isum.0.lcssa, %conv
  %conv6 = fptosi float %div5 to i32
  store i32 %conv6, i32* %iOutput, align 4
  ret void
}

define dso_local i32 @i32_mul_reduction(i32* noalias nocapture readonly %B, i32 %N) local_unnamed_addr #0 {
; CHECK-LABEL: i32_mul_reduction(
; CHECK:       vector.body:
; CHECK-NOT:   @llvm.masked.load
; CHECK-NOT:   @llvm.masked.store
; CHECK:       br i1 %{{.*}}, label {{.*}}, label %vector.body
entry:
  %cmp6 = icmp sgt i32 %N, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.cleanup.loopexit:
  %mul.lcssa = phi i32 [ %mul, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:
  %S.0.lcssa = phi i32 [ 1, %entry ], [ %mul.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %S.0.lcssa

for.body:
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %S.07 = phi i32 [ %mul, %for.body ], [ 1, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %mul = mul nsw i32 %0, %S.07
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

define dso_local i32 @i32_or_reduction(i32* noalias nocapture readonly %B, i32 %N) local_unnamed_addr #0 {
; CHECK-LABEL: i32_or_reduction(
; CHECK:       vector.body:
; CHECK-NOT:   @llvm.masked.load
; CHECK-NOT:   @llvm.masked.store
; CHECK:       br i1 %{{.*}}, label {{.*}}, label %vector.body
entry:
  %cmp6 = icmp sgt i32 %N, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %or.lcssa = phi i32 [ %or, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %S.0.lcssa = phi i32 [ 1, %entry ], [ %or.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %S.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %S.07 = phi i32 [ %or, %for.body ], [ 1, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %or = or i32 %0, %S.07
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

define dso_local i32 @i32_and_reduction(i32* noalias nocapture readonly %A, i32 %N, i32 %S) local_unnamed_addr #0 {
; CHECK-LABEL: i32_and_reduction(
; CHECK:       vector.body:
; CHECK-NOT:   @llvm.masked.load
; CHECK-NOT:   @llvm.masked.store
; CHECK:       br i1 %{{.*}}, label {{.*}}, label %vector.body
entry:
  %cmp5 = icmp sgt i32 %N, 0
  br i1 %cmp5, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %and.lcssa = phi i32 [ %and, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %S.addr.0.lcssa = phi i32 [ %S, %entry ], [ %and.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %S.addr.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.07 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %S.addr.06 = phi i32 [ %and, %for.body ], [ %S, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.07
  %0 = load i32, i32* %arrayidx, align 4
  %and = and i32 %0, %S.addr.06
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

define i32 @i32_smin_reduction(i32* nocapture readonly %x, i32 %n) #0 {
; CHECK-LABEL: i32_smin_reduction(
; CHECK:       vector.body:
; CHECK-NOT:   @llvm.masked.load
; CHECK-NOT:   @llvm.masked.store
; CHECK:       br i1 %{{.*}}, label {{.*}}, label %vector.body
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %r.07 = phi i32 [ %add, %for.body ], [ 2147483647, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %x, i32 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %c = icmp slt i32 %r.07, %0
  %add = select i1 %c, i32 %r.07, i32 %0
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %r.0.lcssa = phi i32 [ 2147483647, %entry ], [ %add, %for.body ]
  ret i32 %r.0.lcssa
}

define i32 @i32_smax_reduction(i32* nocapture readonly %x, i32 %n) #0 {
; CHECK-LABEL: i32_smax_reduction(
; CHECK:       vector.body:
; CHECK-NOT:   @llvm.masked.load
; CHECK-NOT:   @llvm.masked.store
; CHECK:       br i1 %{{.*}}, label {{.*}}, label %vector.body
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %r.07 = phi i32 [ %add, %for.body ], [ -2147483648, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %x, i32 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %c = icmp sgt i32 %r.07, %0
  %add = select i1 %c, i32 %r.07, i32 %0
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %r.0.lcssa = phi i32 [ -2147483648, %entry ], [ %add, %for.body ]
  ret i32 %r.0.lcssa
}

define i32 @i32_umin_reduction(i32* nocapture readonly %x, i32 %n) #0 {
; CHECK-LABEL: i32_umin_reduction(
; CHECK:       vector.body:
; CHECK-NOT:   @llvm.masked.load
; CHECK-NOT:   @llvm.masked.store
; CHECK:       br i1 %{{.*}}, label {{.*}}, label %vector.body
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %r.07 = phi i32 [ %add, %for.body ], [ 4294967295, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %x, i32 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %c = icmp ult i32 %r.07, %0
  %add = select i1 %c, i32 %r.07, i32 %0
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %r.0.lcssa = phi i32 [ 4294967295, %entry ], [ %add, %for.body ]
  ret i32 %r.0.lcssa
}

define i32 @i32_umax_reduction(i32* nocapture readonly %x, i32 %n) #0 {
; CHECK-LABEL: i32_umax_reduction(
; CHECK:       vector.body:
; CHECK-NOT:   @llvm.masked.load
; CHECK-NOT:   @llvm.masked.store
; CHECK:       br i1 %{{.*}}, label {{.*}}, label %vector.body
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %r.07 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %x, i32 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %c = icmp ugt i32 %r.07, %0
  %add = select i1 %c, i32 %r.07, i32 %0
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %r.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %r.0.lcssa
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
