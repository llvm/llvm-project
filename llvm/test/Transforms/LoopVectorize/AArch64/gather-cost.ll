; RUN: opt -loop-vectorize -mtriple=arm64-apple-ios -S -mcpu=cyclone -enable-interleaved-mem-accesses=false < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64-S128"

@kernel = global [512 x float] zeroinitializer, align 16
@kernel2 = global [512 x float] zeroinitializer, align 16
@kernel3 = global [512 x float] zeroinitializer, align 16
@kernel4 = global [512 x float] zeroinitializer, align 16
@src_data = global [1536 x float] zeroinitializer, align 16
@r_ = global i8 0, align 1
@g_ = global i8 0, align 1
@b_ = global i8 0, align 1

; We don't want to vectorize most loops containing gathers because they are
; expensive.
; Make sure we don't vectorize it.
; CHECK-NOT: x float>

define void @_Z4testmm(i64 %size, i64 %offset) {
entry:
  %cmp53 = icmp eq i64 %size, 0
  br i1 %cmp53, label %for.end, label %for.body.lr.ph

for.body.lr.ph:
  br label %for.body

for.body:
  %r.057 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add10, %for.body ]
  %g.056 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add20, %for.body ]
  %v.055 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %b.054 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add30, %for.body ]
  %add = add i64 %v.055, %offset
  %mul = mul i64 %add, 3
  %arrayidx = getelementptr inbounds [1536 x float], [1536 x float]* @src_data, i64 0, i64 %mul
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [512 x float], [512 x float]* @kernel, i64 0, i64 %v.055
  %1 = load float, float* %arrayidx2, align 4
  %mul3 = fmul fast float %0, %1
  %arrayidx4 = getelementptr inbounds [512 x float], [512 x float]* @kernel2, i64 0, i64 %v.055
  %2 = load float, float* %arrayidx4, align 4
  %mul5 = fmul fast float %mul3, %2
  %arrayidx6 = getelementptr inbounds [512 x float], [512 x float]* @kernel3, i64 0, i64 %v.055
  %3 = load float, float* %arrayidx6, align 4
  %mul7 = fmul fast float %mul5, %3
  %arrayidx8 = getelementptr inbounds [512 x float], [512 x float]* @kernel4, i64 0, i64 %v.055
  %4 = load float, float* %arrayidx8, align 4
  %mul9 = fmul fast float %mul7, %4
  %add10 = fadd fast float %r.057, %mul9
  %arrayidx.sum = add i64 %mul, 1
  %arrayidx11 = getelementptr inbounds [1536 x float], [1536 x float]* @src_data, i64 0, i64 %arrayidx.sum
  %5 = load float, float* %arrayidx11, align 4
  %mul13 = fmul fast float %1, %5
  %mul15 = fmul fast float %2, %mul13
  %mul17 = fmul fast float %3, %mul15
  %mul19 = fmul fast float %4, %mul17
  %add20 = fadd fast float %g.056, %mul19
  %arrayidx.sum52 = add i64 %mul, 2
  %arrayidx21 = getelementptr inbounds [1536 x float], [1536 x float]* @src_data, i64 0, i64 %arrayidx.sum52
  %6 = load float, float* %arrayidx21, align 4
  %mul23 = fmul fast float %1, %6
  %mul25 = fmul fast float %2, %mul23
  %mul27 = fmul fast float %3, %mul25
  %mul29 = fmul fast float %4, %mul27
  %add30 = fadd fast float %b.054, %mul29
  %inc = add i64 %v.055, 1
  %exitcond = icmp ne i64 %inc, %size
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:
  %add30.lcssa = phi float [ %add30, %for.body ]
  %add20.lcssa = phi float [ %add20, %for.body ]
  %add10.lcssa = phi float [ %add10, %for.body ]
  %phitmp = fptoui float %add10.lcssa to i8
  %phitmp60 = fptoui float %add20.lcssa to i8
  %phitmp61 = fptoui float %add30.lcssa to i8
  br label %for.end

for.end:
  %r.0.lcssa = phi i8 [ %phitmp, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  %g.0.lcssa = phi i8 [ %phitmp60, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  %b.0.lcssa = phi i8 [ %phitmp61, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  store i8 %r.0.lcssa, i8* @r_, align 1
  store i8 %g.0.lcssa, i8* @g_, align 1
  store i8 %b.0.lcssa, i8* @b_, align 1
  ret void
}
