; PR34438
; Loop has a short trip count of 8 iterations. It should be vectorized because no runtime checks or tail loop are necessary.
; Two cases tested AVX (MaxVF=8 = TripCount) and AVX512 (MaxVF=16 > TripCount)

; RUN: opt < %s -loop-vectorize -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx -S | FileCheck %s
; RUN: opt < %s -loop-vectorize -mtriple=x86_64-apple-macosx10.8.0 -mcpu=skylake-avx512 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define void @small_tc(float* noalias nocapture %A, float* noalias nocapture readonly %B) {
; CHECK-LABEL: @small_tc
; CHECK:    load <8 x float>, <8 x float>*
; CHECK:    fadd fast <8 x float>
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %B, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4, !llvm.access.group !5
  %arrayidx2 = getelementptr inbounds float, float* %A, i64 %indvars.iv
  %1 = load float, float* %arrayidx2, align 4, !llvm.access.group !5
  %add = fadd fast float %0, %1
  store float %add, float* %arrayidx2, align 4, !llvm.access.group !5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 8
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}

!3 = !{!3, !{!"llvm.loop.parallel_accesses", !5}}
!4 = !{!4}
!5 = distinct !{}
