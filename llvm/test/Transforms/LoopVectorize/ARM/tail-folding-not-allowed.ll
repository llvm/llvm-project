; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp \
; RUN:   -tail-predication=enabled -loop-vectorize -S < %s | \
; RUN:   FileCheck %s

define void @trunc_not_allowed_different_vec_elemns(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i16* noalias nocapture %D) #0 {
; CHECK-LABEL: trunc_not_allowed_different_vec_elemns(
; CHECK:       vector.body:
; CHECK-NOT:   llvm.masked.load
; CHECK-NOT:   llvm.masked.store
; CHECK:       br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.021 = phi i32 [ 0, %entry ], [ %add9, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.021
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.021
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.021
  store i32 %add, i32* %arrayidx2, align 4
  %add.tr = trunc i32 %add to i16
  %conv7 = shl i16 %add.tr, 1
  %arrayidx8 = getelementptr inbounds i16, i16* %D, i32 %i.021
  store i16 %conv7, i16* %arrayidx8, align 2
  %add9 = add nuw nsw i32 %i.021, 1
  %exitcond = icmp eq i32 %add9, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @unsupported_i64_type(i64* noalias nocapture %A, i64* noalias nocapture readonly %B, i64* noalias nocapture readonly %C) #0 {
; CHECK-LABEL: unsupported_i64_type(
; CHECK-NOT:   vector.body:
; CHECK-NOT:   llvm.masked.load
; CHECK-NOT:   llvm.masked.store
; CHECK:       for.body:
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %B, i32 %i.09
  %0 = load i64, i64* %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, i64* %C, i32 %i.09
  %1 = load i64, i64* %arrayidx1, align 8
  %add = add nsw i64 %1, %0
  %arrayidx2 = getelementptr inbounds i64, i64* %A, i32 %i.09
  store i64 %add, i64* %arrayidx2, align 8
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @narrowing_load_not_allowed(i8* noalias nocapture %A, i8* noalias nocapture readonly %B, i16* noalias nocapture readonly %C) #0 {
; CHECK-LABEL: narrowing_load_not_allowed(
; CHECK:       vector.body:
; CHECK-NOT:   llvm.masked.load
; CHECK-NOT:   llvm.masked.store
; CHECK:       br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.012 = phi i32 [ 0, %entry ], [ %add6, %for.body ]
  %arrayidx = getelementptr inbounds i16, i16* %C, i32 %i.012
  %0 = load i16, i16* %arrayidx, align 2
  %arrayidx1 = getelementptr inbounds i8, i8* %B, i32 %i.012
  %1 = load i8, i8* %arrayidx1, align 1
  %conv3 = trunc i16 %0 to i8
  %add = add i8 %1, %conv3
  %arrayidx5 = getelementptr inbounds i8, i8* %A, i32 %i.012
  store i8 %add, i8* %arrayidx5, align 1
  %add6 = add nuw nsw i32 %i.012, 1
  %exitcond = icmp eq i32 %add6, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; This is a trunc not connected to a store, so we don't allow this.
; TODO: this is conservative, because the trunc is only used in the
; loop control statements, and thus not affecting element sizes, so
; we could allow this case.
;
define void @trunc_not_allowed(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:  trunc_not_allowed(
; CHECK:        vector.body:
; CHECK-NOT:    llvm.masked.load
; CHECK-NOT:    llvm.masked.store
; CHECK:        br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1

  %add.iv = trunc i32 %add3 to i16

  %exitcond = icmp eq i16 %add.iv, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Test directions for array indices i and N-1. I.e. check strides 1 and -1, and
; force vectorisation with a loop hint.
;
define void @strides_different_direction(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i32 %N) #0 {
; CHECK-LABEL: strides_different_direction(
; CHECK:       vector.body:
; CHECK-NOT:   llvm.masked.load
; CHECK-NOT:   llvm.masked.store
; CHECK:       br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %sub = sub nsw i32 %N, %i.09
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %sub
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !10
}

define void @too_many_loop_blocks(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL: too_many_loop_blocks(
; CHECK:       vector.body:
; CHECK-NOT:   llvm.masked.load
; CHECK-NOT:   llvm.masked.store
; CHECK:       br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %loopincr ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  br label %loopincr

loopincr:
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @double(double* noalias nocapture %A, double* noalias nocapture readonly %B, double* noalias nocapture readonly %C) #0 {
; CHECK-LABEL: double(
; CHECK:       for.body:
; CHECK-NOT:   vector.body:
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %B, i32 %i.09
  %0 = load double, double* %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds double, double* %C, i32 %i.09
  %1 = load double, double* %arrayidx1, align 8
  %add = fadd fast double %1, %0
  %arrayidx2 = getelementptr inbounds double, double* %A, i32 %i.09
  store double %add, double* %arrayidx2, align 8
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @fptrunc_not_allowed(float* noalias nocapture %A, float* noalias nocapture readonly %B, float* noalias nocapture readonly %C, half* noalias nocapture %D) #0 {
; CHECK-LABEL: fptrunc_not_allowed(
; CHECK-NOT:   vector.body:
; CHECK-NOT:   llvm.masked.load
; CHECK-NOT:   llvm.masked.store
; CHECK:       br i1 %{{.*}}, label %{{.*}}, label %for.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.017 = phi i32 [ 0, %entry ], [ %add6, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %B, i32 %i.017
  %0 = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %C, i32 %i.017
  %1 = load float, float* %arrayidx1, align 4
  %add = fadd fast float %1, %0
  %arrayidx2 = getelementptr inbounds float, float* %A, i32 %i.017
  store float %add, float* %arrayidx2, align 4
  %conv = fptrunc float %add to half
  %factor = fmul fast half %conv, 0xH4000
  %arrayidx5 = getelementptr inbounds half, half* %D, i32 %i.017
  store half %factor, half* %arrayidx5, align 2
  %add6 = add nuw nsw i32 %i.017, 1
  %exitcond = icmp eq i32 %add6, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; This is a select which isn't a max or min (it isn't live-out), that we don't
; want to tail-fold. Because this select will result in some mov lanes,
; which aren't supported by the lowoverhead loop pass, causing the tail-predication
; to be reverted which is expensive and what we would like to avoid.
;
define dso_local void @select_not_allowed(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i32 %N, i32* noalias nocapture readonly %Cond) {
entry:
  %cmp10 = icmp sgt i32 %N, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.011 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %Cond, i32 %i.011
  %0 = load i32, i32* %arrayidx, align 4
  %tobool.not = icmp eq i32 %0, 0
  %C.B = select i1 %tobool.not, i32* %C, i32* %B
  %cond.in = getelementptr inbounds i32, i32* %C.B, i32 %i.011
  %cond = load i32, i32* %cond.in, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i32 %i.011
  store i32 %cond, i32* %arrayidx3, align 4
  %inc = add nuw nsw i32 %i.011, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
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

!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.vectorize.width", i32 4}
