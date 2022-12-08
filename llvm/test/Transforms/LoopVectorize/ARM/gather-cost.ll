; RUN: opt -passes=loop-vectorize -mtriple=thumbv7s-apple-ios6.0.0 -S -enable-interleaved-mem-accesses=false < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"

@kernel = global [512 x float] zeroinitializer, align 4
@kernel2 = global [512 x float] zeroinitializer, align 4
@kernel3 = global [512 x float] zeroinitializer, align 4
@kernel4 = global [512 x float] zeroinitializer, align 4
@src_data = global [1536 x float] zeroinitializer, align 4
@r_ = global i8 0, align 4
@g_ = global i8 0, align 4
@b_ = global i8 0, align 4

; We don't want to vectorize most loops containing gathers because they are
; expensive. This function represents a point where vectorization starts to
; become beneficial.
; Make sure we are conservative and don't vectorize it.
; CHECK-NOT: <2 x float>
; CHECK-NOT: <4 x float>

define void @_Z4testmm(i32 %size, i32 %offset) {
entry:
  %cmp53 = icmp eq i32 %size, 0
  br i1 %cmp53, label %for.end, label %for.body.lr.ph

for.body.lr.ph:
  br label %for.body

for.body:
  %r.057 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add10, %for.body ]
  %g.056 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add20, %for.body ]
  %v.055 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %b.054 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add30, %for.body ]
  %add = add i32 %v.055, %offset
  %mul = mul i32 %add, 3
  %arrayidx = getelementptr inbounds [1536 x float], [1536 x float]* @src_data, i32 0, i32 %mul
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [512 x float], [512 x float]* @kernel, i32 0, i32 %v.055
  %1 = load float, float* %arrayidx2, align 4
  %mul3 = fmul fast float %0, %1
  %arrayidx4 = getelementptr inbounds [512 x float], [512 x float]* @kernel2, i32 0, i32 %v.055
  %2 = load float, float* %arrayidx4, align 4
  %mul5 = fmul fast float %mul3, %2
  %arrayidx6 = getelementptr inbounds [512 x float], [512 x float]* @kernel3, i32 0, i32 %v.055
  %3 = load float, float* %arrayidx6, align 4
  %mul7 = fmul fast float %mul5, %3
  %arrayidx8 = getelementptr inbounds [512 x float], [512 x float]* @kernel4, i32 0, i32 %v.055
  %4 = load float, float* %arrayidx8, align 4
  %mul9 = fmul fast float %mul7, %4
  %add10 = fadd fast float %r.057, %mul9
  %arrayidx.sum = add i32 %mul, 1
  %arrayidx11 = getelementptr inbounds [1536 x float], [1536 x float]* @src_data, i32 0, i32 %arrayidx.sum
  %5 = load float, float* %arrayidx11, align 4
  %mul13 = fmul fast float %1, %5
  %mul15 = fmul fast float %2, %mul13
  %mul17 = fmul fast float %3, %mul15
  %mul19 = fmul fast float %4, %mul17
  %add20 = fadd fast float %g.056, %mul19
  %arrayidx.sum52 = add i32 %mul, 2
  %arrayidx21 = getelementptr inbounds [1536 x float], [1536 x float]* @src_data, i32 0, i32 %arrayidx.sum52
  %6 = load float, float* %arrayidx21, align 4
  %mul23 = fmul fast float %1, %6
  %mul25 = fmul fast float %2, %mul23
  %mul27 = fmul fast float %3, %mul25
  %mul29 = fmul fast float %4, %mul27
  %add30 = fadd fast float %b.054, %mul29
  %inc = add i32 %v.055, 1
  %exitcond = icmp ne i32 %inc, %size
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
  store i8 %r.0.lcssa, i8* @r_, align 4
  store i8 %g.0.lcssa, i8* @g_, align 4
  store i8 %b.0.lcssa, i8* @b_, align 4
  ret void
}
