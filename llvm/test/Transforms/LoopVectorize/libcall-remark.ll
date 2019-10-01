; RUN: opt -S -loop-vectorize < %s 2>&1 -pass-remarks-analysis=.* | FileCheck %s

; Test the optimization remark emitter for recognition 
; of a mathlib function vs. an arbitrary function.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"
@data = external local_unnamed_addr global [32768 x float], align 16

; CHECK: loop not vectorized: library call cannot be vectorized

define void @libcall_blocks_vectorization() {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [32768 x float], [32768 x float]* @data, i64 0, i64 %indvars.iv
  %t0 = load float, float* %arrayidx, align 4
  %sqrtf = tail call float @sqrtf(float %t0)
  store float %sqrtf, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 32768
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK: loop not vectorized: call instruction cannot be vectorized

define void @arbitrary_call_blocks_vectorization() {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [32768 x float], [32768 x float]* @data, i64 0, i64 %indvars.iv
  %t0 = load float, float* %arrayidx, align 4
  %sqrtf = tail call float @arbitrary(float %t0)
  store float %sqrtf, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 32768
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

declare float @sqrtf(float)
declare float @arbitrary(float)

