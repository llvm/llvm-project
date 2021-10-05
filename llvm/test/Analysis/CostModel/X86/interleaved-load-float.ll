; REQUIRES: asserts
; RUN: opt -S -loop-vectorize -debug-only=loop-vectorize -mcpu=skylake %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@src = common local_unnamed_addr global [120 x float] zeroinitializer, align 4
@dst = common local_unnamed_addr global [120 x float] zeroinitializer, align 4

; Function Attrs: norecurse nounwind
define void @stride8(float %k, i32 %width_) {
entry:

; CHECK: Found an estimated cost of 48 for VF 8 For instruction:   %0 = load float

  %cmp72 = icmp sgt i32 %width_, 0
  br i1 %cmp72, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.073 = phi i32 [ 0, %for.body.lr.ph ], [ %add46, %for.body ]
  %arrayidx = getelementptr inbounds [120 x float], [120 x float]* @src, i32 0, i32 %i.073
  %0 = load float, float* %arrayidx, align 4
  %mul = fmul fast float %0, %k
  %arrayidx2 = getelementptr inbounds [120 x float], [120 x float]* @dst, i32 0, i32 %i.073
  %1 = load float, float* %arrayidx2, align 4
  %add3 = fadd fast float %1, %mul
  store float %add3, float* %arrayidx2, align 4
  %add4 = or i32 %i.073, 1
  %arrayidx5 = getelementptr inbounds [120 x float], [120 x float]* @src, i32 0, i32 %add4
  %2 = load float, float* %arrayidx5, align 4
  %mul6 = fmul fast float %2, %k
  %arrayidx8 = getelementptr inbounds [120 x float], [120 x float]* @dst, i32 0, i32 %add4
  %3 = load float, float* %arrayidx8, align 4
  %add9 = fadd fast float %3, %mul6
  store float %add9, float* %arrayidx8, align 4
  %add10 = or i32 %i.073, 2
  %arrayidx11 = getelementptr inbounds [120 x float], [120 x float]* @src, i32 0, i32 %add10
  %4 = load float, float* %arrayidx11, align 4
  %mul12 = fmul fast float %4, %k
  %arrayidx14 = getelementptr inbounds [120 x float], [120 x float]* @dst, i32 0, i32 %add10
  %5 = load float, float* %arrayidx14, align 4
  %add15 = fadd fast float %5, %mul12
  store float %add15, float* %arrayidx14, align 4
  %add16 = or i32 %i.073, 3
  %arrayidx17 = getelementptr inbounds [120 x float], [120 x float]* @src, i32 0, i32 %add16
  %6 = load float, float* %arrayidx17, align 4
  %mul18 = fmul fast float %6, %k
  %arrayidx20 = getelementptr inbounds [120 x float], [120 x float]* @dst, i32 0, i32 %add16
  %7 = load float, float* %arrayidx20, align 4
  %add21 = fadd fast float %7, %mul18
  store float %add21, float* %arrayidx20, align 4
  %add22 = or i32 %i.073, 4
  %arrayidx23 = getelementptr inbounds [120 x float], [120 x float]* @src, i32 0, i32 %add22
  %8 = load float, float* %arrayidx23, align 4
  %mul24 = fmul fast float %8, %k
  %arrayidx26 = getelementptr inbounds [120 x float], [120 x float]* @dst, i32 0, i32 %add22
  %9 = load float, float* %arrayidx26, align 4
  %add27 = fadd fast float %9, %mul24
  store float %add27, float* %arrayidx26, align 4
  %add28 = or i32 %i.073, 5
  %arrayidx29 = getelementptr inbounds [120 x float], [120 x float]* @src, i32 0, i32 %add28
  %10 = load float, float* %arrayidx29, align 4
  %mul30 = fmul fast float %10, %k
  %arrayidx32 = getelementptr inbounds [120 x float], [120 x float]* @dst, i32 0, i32 %add28
  %11 = load float, float* %arrayidx32, align 4
  %add33 = fadd fast float %11, %mul30
  store float %add33, float* %arrayidx32, align 4
  %add34 = or i32 %i.073, 6
  %arrayidx35 = getelementptr inbounds [120 x float], [120 x float]* @src, i32 0, i32 %add34
  %12 = load float, float* %arrayidx35, align 4
  %mul36 = fmul fast float %12, %k
  %arrayidx38 = getelementptr inbounds [120 x float], [120 x float]* @dst, i32 0, i32 %add34
  %13 = load float, float* %arrayidx38, align 4
  %add39 = fadd fast float %13, %mul36
  store float %add39, float* %arrayidx38, align 4
  %add40 = or i32 %i.073, 7
  %arrayidx41 = getelementptr inbounds [120 x float], [120 x float]* @src, i32 0, i32 %add40
  %14 = load float, float* %arrayidx41, align 4
  %mul42 = fmul fast float %14, %k
  %arrayidx44 = getelementptr inbounds [120 x float], [120 x float]* @dst, i32 0, i32 %add40
  %15 = load float, float* %arrayidx44, align 4
  %add45 = fadd fast float %15, %mul42
  store float %add45, float* %arrayidx44, align 4
  %add46 = add nuw nsw i32 %i.073, 8
  %cmp = icmp slt i32 %add46, %width_
  br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit
}
