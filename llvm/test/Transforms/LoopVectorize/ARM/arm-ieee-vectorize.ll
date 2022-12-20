; RUN: opt -mtriple armv7-linux-gnueabihf -passes=loop-vectorize -S %s -debug-only=loop-vectorize -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=LINUX
; RUN: opt -mtriple armv8-linux-gnu -passes=loop-vectorize -S %s -debug-only=loop-vectorize -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=LINUX
; RUN: opt -mtriple armv8.1.m-none-eabi -mattr=+mve.fp -passes=loop-vectorize -S %s -debug-only=loop-vectorize -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=MVE
; RUN: opt -mtriple armv7-unknwon-darwin -passes=loop-vectorize -S %s -debug-only=loop-vectorize -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=DARWIN
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

; Testing the ability of the loop vectorizer to tell when SIMD is safe or not
; regarding IEEE 754 standard.
; On Linux, we only want the vectorizer to work when -ffast-math flag is set,
; because NEON is not IEEE compliant.
; Darwin, on the other hand, doesn't support subnormals, and all optimizations
; are allowed, even without -ffast-math.

; Integer loops are always vectorizeable
; CHECK: Checking a loop in 'sumi'
; CHECK: We can vectorize this loop!
define void @sumi(ptr noalias nocapture readonly %A, ptr noalias nocapture readonly %B, ptr noalias nocapture %C, i32 %N) {
entry:
  %cmp5 = icmp eq i32 %N, 0
  br i1 %cmp5, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i.06
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %B, i32 %i.06
  %1 = load i32, ptr %arrayidx1, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, ptr %C, i32 %i.06
  store i32 %mul, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; Floating-point loops need fast-math to be vectorizeable
; LINUX: Checking a loop in 'sumf'
; LINUX: Potentially unsafe FP op prevents vectorization
; MVE: Checking a loop in 'sumf'
; MVE: We can vectorize this loop!
; DARWIN: Checking a loop in 'sumf'
; DARWIN: We can vectorize this loop!
define void @sumf(ptr noalias nocapture readonly %A, ptr noalias nocapture readonly %B, ptr noalias nocapture %C, i32 %N) {
entry:
  %cmp5 = icmp eq i32 %N, 0
  br i1 %cmp5, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, ptr %A, i32 %i.06
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %B, i32 %i.06
  %1 = load float, ptr %arrayidx1, align 4
  %mul = fmul float %0, %1
  %arrayidx2 = getelementptr inbounds float, ptr %C, i32 %i.06
  store float %mul, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; Integer loops are always vectorizeable
; CHECK: Checking a loop in 'redi'
; CHECK: We can vectorize this loop!
define i32 @redi(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, i32 %N) {
entry:
  %cmp5 = icmp eq i32 %N, 0
  br i1 %cmp5, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.07 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %Red.06 = phi i32 [ %add, %for.body ], [ undef, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %i.07
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %b, i32 %i.07
  %1 = load i32, ptr %arrayidx1, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, %Red.06
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %Red.0.lcssa = phi i32 [ undef, %entry ], [ %add.lcssa, %for.end.loopexit ]
  ret i32 %Red.0.lcssa
}

; Floating-point loops need fast-math to be vectorizeable
; LINUX: Checking a loop in 'redf'
; LINUX: Potentially unsafe FP op prevents vectorization
; MVE: Checking a loop in 'redf'
; MVE: We can vectorize this loop!
; DARWIN: Checking a loop in 'redf'
; DARWIN: We can vectorize this loop!
define float @redf(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, i32 %N) {
entry:
  %cmp5 = icmp eq i32 %N, 0
  br i1 %cmp5, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.07 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %Red.06 = phi float [ %add, %for.body ], [ undef, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, ptr %a, i32 %i.07
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %b, i32 %i.07
  %1 = load float, ptr %arrayidx1, align 4
  %mul = fmul float %0, %1
  %add = fadd float %Red.06, %mul
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  %add.lcssa = phi float [ %add, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %Red.0.lcssa = phi float [ undef, %entry ], [ %add.lcssa, %for.end.loopexit ]
  ret float %Red.0.lcssa
}

; Make sure calls that turn into builtins are also covered
; LINUX: Checking a loop in 'fabs'
; LINUX: Potentially unsafe FP op prevents vectorization
; DARWIN: Checking a loop in 'fabs'
; DARWIN: We can vectorize this loop!
define void @fabs(ptr noalias nocapture readonly %A, ptr noalias nocapture readonly %B, ptr noalias nocapture %C, i32 %N) {
entry:
  %cmp10 = icmp eq i32 %N, 0
  br i1 %cmp10, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.011 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %A, i32 %i.011
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %B, i32 %i.011
  %1 = load float, ptr %arrayidx1, align 4
  %fabsf = tail call float @fabsf(float %1) #1
  %conv3 = fmul float %0, %fabsf
  %arrayidx4 = getelementptr inbounds float, ptr %C, i32 %i.011
  store float %conv3, ptr %arrayidx4, align 4
  %inc = add nuw nsw i32 %i.011, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Integer loops are always vectorizeable
; CHECK: Checking a loop in 'sumi_fast'
; CHECK: We can vectorize this loop!
define void @sumi_fast(ptr noalias nocapture readonly %A, ptr noalias nocapture readonly %B, ptr noalias nocapture %C, i32 %N) {
entry:
  %cmp5 = icmp eq i32 %N, 0
  br i1 %cmp5, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i.06
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %B, i32 %i.06
  %1 = load i32, ptr %arrayidx1, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, ptr %C, i32 %i.06
  store i32 %mul, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; Floating-point loops can be vectorizeable with fast-math
; CHECK: Checking a loop in 'sumf_fast'
; CHECK: We can vectorize this loop!
define void @sumf_fast(ptr noalias nocapture readonly %A, ptr noalias nocapture readonly %B, ptr noalias nocapture %C, i32 %N) {
entry:
  %cmp5 = icmp eq i32 %N, 0
  br i1 %cmp5, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, ptr %A, i32 %i.06
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %B, i32 %i.06
  %1 = load float, ptr %arrayidx1, align 4
  %mul = fmul fast float %1, %0
  %arrayidx2 = getelementptr inbounds float, ptr %C, i32 %i.06
  store float %mul, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; Integer loops are always vectorizeable
; CHECK: Checking a loop in 'redi_fast'
; CHECK: We can vectorize this loop!
define i32 @redi_fast(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, i32 %N) {
entry:
  %cmp5 = icmp eq i32 %N, 0
  br i1 %cmp5, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.07 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %Red.06 = phi i32 [ %add, %for.body ], [ undef, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %i.07
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %b, i32 %i.07
  %1 = load i32, ptr %arrayidx1, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, %Red.06
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %Red.0.lcssa = phi i32 [ undef, %entry ], [ %add.lcssa, %for.end.loopexit ]
  ret i32 %Red.0.lcssa
}

; Floating-point loops can be vectorizeable with fast-math
; CHECK: Checking a loop in 'redf_fast'
; CHECK: We can vectorize this loop!
define float @redf_fast(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, i32 %N) {
entry:
  %cmp5 = icmp eq i32 %N, 0
  br i1 %cmp5, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.07 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %Red.06 = phi float [ %add, %for.body ], [ undef, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, ptr %a, i32 %i.07
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %b, i32 %i.07
  %1 = load float, ptr %arrayidx1, align 4
  %mul = fmul fast float %1, %0
  %add = fadd fast float %mul, %Red.06
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  %add.lcssa = phi float [ %add, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %Red.0.lcssa = phi float [ undef, %entry ], [ %add.lcssa, %for.end.loopexit ]
  ret float %Red.0.lcssa
}

; Make sure calls that turn into builtins are also covered
; CHECK: Checking a loop in 'fabs_fast'
; CHECK: We can vectorize this loop!
define void @fabs_fast(ptr noalias nocapture readonly %A, ptr noalias nocapture readonly %B, ptr noalias nocapture %C, i32 %N) {
entry:
  %cmp10 = icmp eq i32 %N, 0
  br i1 %cmp10, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.011 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %A, i32 %i.011
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %B, i32 %i.011
  %1 = load float, ptr %arrayidx1, align 4
  %fabsf = tail call fast float @fabsf(float %1) #2
  %conv3 = fmul fast float %fabsf, %0
  %arrayidx4 = getelementptr inbounds float, ptr %C, i32 %i.011
  store float %conv3, ptr %arrayidx4, align 4
  %inc = add nuw nsw i32 %i.011, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @fabsf(float)

attributes #1 = { nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a8" "target-features"="+dsp,+neon,+vfp3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a8" "target-features"="+dsp,+neon,+vfp3" "unsafe-fp-math"="true" "use-soft-float"="false" }
