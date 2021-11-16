; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp -loop-vectorize -tail-predication=enabled -S < %s | \
; RUN:  FileCheck %s

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

; Test that ARMTTIImpl::preferPredicateOverEpilogue triggers tail-folding.

define dso_local void @f1(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i32 %N) {
; CHECK-LABEL: f1(
; CHECK:       entry:
; CHECK:       @llvm.get.active.lane.mask
; CHECK:       }
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
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}

define dso_local void @f32_reduction(float* nocapture readonly %Input, i32 %N, float* nocapture %Output) {
; CHECK-LABEL: f32_reduction(
; CHECK:       vector.body:
; CHECK:       @llvm.masked.load
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

define dso_local void @f16_reduction(half* nocapture readonly %Input, i32 %N, half* nocapture %Output) {
; CHECK-LABEL: f16_reduction(
; CHECK:       vector.body:
; CHECK:       @llvm.masked.load
; CHECK:       br i1 %{{.*}}, label {{.*}}, label %vector.body
entry:
  %cmp6 = icmp eq i32 %N, 0
  br i1 %cmp6, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %blkCnt.09 = phi i32 [ %dec, %while.body ], [ %N, %while.body.preheader ]
  %sum.08 = phi half [ %add, %while.body ], [ 0.000000e+00, %while.body.preheader ]
  %Input.addr.07 = phi half* [ %incdec.ptr, %while.body ], [ %Input, %while.body.preheader ]
  %incdec.ptr = getelementptr inbounds half, half* %Input.addr.07, i32 1
  %0 = load half, half* %Input.addr.07, align 2
  %add = fadd fast half %0, %sum.08
  %dec = add i32 %blkCnt.09, -1
  %cmp = icmp eq i32 %dec, 0
  br i1 %cmp, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  %add.lcssa = phi half [ %add, %while.body ]
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %sum.0.lcssa = phi half [ 0.000000e+00, %entry ], [ %add.lcssa, %while.end.loopexit ]
  %conv = uitofp i32 %N to half
  %div = fdiv fast half %sum.0.lcssa, %conv
  store half %div, half* %Output, align 2
  ret void
}

define dso_local void @mixed_f32_i32_reduction(float* nocapture readonly %fInput, i32* nocapture readonly %iInput, i32 %N, float* nocapture %fOutput, i32* nocapture %iOutput) {
; CHECK-LABEL: mixed_f32_i32_reduction(
; CHECK:       vector.body:
; CHECK:       @llvm.masked.load
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

define dso_local i32 @i32_mul_reduction(i32* noalias nocapture readonly %B, i32 %N) {
; CHECK-LABEL: i32_mul_reduction(
; CHECK:       vector.body:
; CHECK:       @llvm.masked.load
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

define dso_local i32 @i32_or_reduction(i32* noalias nocapture readonly %B, i32 %N) {
; CHECK-LABEL: i32_or_reduction(
; CHECK:       vector.body:
; CHECK:       @llvm.masked.load
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

define dso_local i32 @i32_and_reduction(i32* noalias nocapture readonly %A, i32 %N, i32 %S) {
; CHECK-LABEL: i32_and_reduction(
; CHECK:       vector.body:
; CHECK:       @llvm.masked.load
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
