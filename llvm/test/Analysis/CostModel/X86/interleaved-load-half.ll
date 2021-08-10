; RUN: opt -S -loop-vectorize -debug-only=loop-vectorize -mattr=avx512fp16 %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@src = common local_unnamed_addr global [120 x half] zeroinitializer, align 4
@dst = common local_unnamed_addr global [120 x half] zeroinitializer, align 4

; Function Attrs: norecurse nounwind
define void @stride8(half %k, i32 %width_) {
entry:

; CHECK: Found an estimated cost of 148 for VF 32 For instruction:   %0 = load half

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
  %arrayidx = getelementptr inbounds [120 x half], [120 x half]* @src, i32 0, i32 %i.073
  %0 = load half, half* %arrayidx, align 4
  %mul = fmul fast half %0, %k
  %arrayidx2 = getelementptr inbounds [120 x half], [120 x half]* @dst, i32 0, i32 %i.073
  %1 = load half, half* %arrayidx2, align 4
  %add3 = fadd fast half %1, %mul
  store half %add3, half* %arrayidx2, align 4
  %add4 = or i32 %i.073, 1
  %arrayidx5 = getelementptr inbounds [120 x half], [120 x half]* @src, i32 0, i32 %add4
  %2 = load half, half* %arrayidx5, align 4
  %mul6 = fmul fast half %2, %k
  %arrayidx8 = getelementptr inbounds [120 x half], [120 x half]* @dst, i32 0, i32 %add4
  %3 = load half, half* %arrayidx8, align 4
  %add9 = fadd fast half %3, %mul6
  store half %add9, half* %arrayidx8, align 4
  %add10 = or i32 %i.073, 2
  %arrayidx11 = getelementptr inbounds [120 x half], [120 x half]* @src, i32 0, i32 %add10
  %4 = load half, half* %arrayidx11, align 4
  %mul12 = fmul fast half %4, %k
  %arrayidx14 = getelementptr inbounds [120 x half], [120 x half]* @dst, i32 0, i32 %add10
  %5 = load half, half* %arrayidx14, align 4
  %add15 = fadd fast half %5, %mul12
  store half %add15, half* %arrayidx14, align 4
  %add16 = or i32 %i.073, 3
  %arrayidx17 = getelementptr inbounds [120 x half], [120 x half]* @src, i32 0, i32 %add16
  %6 = load half, half* %arrayidx17, align 4
  %mul18 = fmul fast half %6, %k
  %arrayidx20 = getelementptr inbounds [120 x half], [120 x half]* @dst, i32 0, i32 %add16
  %7 = load half, half* %arrayidx20, align 4
  %add21 = fadd fast half %7, %mul18
  store half %add21, half* %arrayidx20, align 4
  %add22 = or i32 %i.073, 4
  %arrayidx23 = getelementptr inbounds [120 x half], [120 x half]* @src, i32 0, i32 %add22
  %8 = load half, half* %arrayidx23, align 4
  %mul24 = fmul fast half %8, %k
  %arrayidx26 = getelementptr inbounds [120 x half], [120 x half]* @dst, i32 0, i32 %add22
  %9 = load half, half* %arrayidx26, align 4
  %add27 = fadd fast half %9, %mul24
  store half %add27, half* %arrayidx26, align 4
  %add28 = or i32 %i.073, 5
  %arrayidx29 = getelementptr inbounds [120 x half], [120 x half]* @src, i32 0, i32 %add28
  %10 = load half, half* %arrayidx29, align 4
  %mul30 = fmul fast half %10, %k
  %arrayidx32 = getelementptr inbounds [120 x half], [120 x half]* @dst, i32 0, i32 %add28
  %11 = load half, half* %arrayidx32, align 4
  %add33 = fadd fast half %11, %mul30
  store half %add33, half* %arrayidx32, align 4
  %add34 = or i32 %i.073, 6
  %arrayidx35 = getelementptr inbounds [120 x half], [120 x half]* @src, i32 0, i32 %add34
  %12 = load half, half* %arrayidx35, align 4
  %mul36 = fmul fast half %12, %k
  %arrayidx38 = getelementptr inbounds [120 x half], [120 x half]* @dst, i32 0, i32 %add34
  %13 = load half, half* %arrayidx38, align 4
  %add39 = fadd fast half %13, %mul36
  store half %add39, half* %arrayidx38, align 4
  %add40 = or i32 %i.073, 7
  %arrayidx41 = getelementptr inbounds [120 x half], [120 x half]* @src, i32 0, i32 %add40
  %14 = load half, half* %arrayidx41, align 4
  %mul42 = fmul fast half %14, %k
  %arrayidx44 = getelementptr inbounds [120 x half], [120 x half]* @dst, i32 0, i32 %add40
  %15 = load half, half* %arrayidx44, align 4
  %add45 = fadd fast half %15, %mul42
  store half %add45, half* %arrayidx44, align 4
  %add46 = add nuw nsw i32 %i.073, 8
  %cmp = icmp slt i32 %add46, %width_
  br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit
}

; Function Attrs: norecurse nounwind
define void @stride3(half %k, i32 %width_) {
entry:

; CHECK: Found an estimated cost of 18 for VF 32 For instruction:   %0 = load half

  %cmp27 = icmp sgt i32 %width_, 0
  br i1 %cmp27, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.028 = phi i32 [ 0, %for.body.lr.ph ], [ %add16, %for.body ]
  %arrayidx = getelementptr inbounds [120 x half], [120 x half]* @src, i32 0, i32 %i.028
  %0 = load half, half* %arrayidx, align 4
  %mul = fmul fast half %0, %k
  %arrayidx2 = getelementptr inbounds [120 x half], [120 x half]* @dst, i32 0, i32 %i.028
  %1 = load half, half* %arrayidx2, align 4
  %add3 = fadd fast half %1, %mul
  store half %add3, half* %arrayidx2, align 4
  %add4 = add nuw nsw i32 %i.028, 1
  %arrayidx5 = getelementptr inbounds [120 x half], [120 x half]* @src, i32 0, i32 %add4
  %2 = load half, half* %arrayidx5, align 4
  %mul6 = fmul fast half %2, %k
  %arrayidx8 = getelementptr inbounds [120 x half], [120 x half]* @dst, i32 0, i32 %add4
  %3 = load half, half* %arrayidx8, align 4
  %add9 = fadd fast half %3, %mul6
  store half %add9, half* %arrayidx8, align 4
  %add10 = add nuw nsw i32 %i.028, 2
  %arrayidx11 = getelementptr inbounds [120 x half], [120 x half]* @src, i32 0, i32 %add10
  %4 = load half, half* %arrayidx11, align 4
  %mul12 = fmul fast half %4, %k
  %arrayidx14 = getelementptr inbounds [120 x half], [120 x half]* @dst, i32 0, i32 %add10
  %5 = load half, half* %arrayidx14, align 4
  %add15 = fadd fast half %5, %mul12
  store half %add15, half* %arrayidx14, align 4
  %add16 = add nuw nsw i32 %i.028, 3
  %cmp = icmp slt i32 %add16, %width_
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

