; RUN: opt < %s -passes=loop-vectorize -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx -debug-only=loop-vectorize -stats -S 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK: LV: Loop hints: force=enabled
; CHECK: LV: Loop hints: force=?
; No more loops in the module
; CHECK-NOT: LV: Loop hints: force=
; CHECK: 2 loop-vectorize               - Number of loops analyzed for vectorization
; CHECK: 1 loop-vectorize               - Number of loops vectorized

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

;
; The source code for the test:
;
; #include <math.h>
; void foo(ptr restrict A, ptr restrict B)
; {
;   for (int i = 0; i < 1000; i+=2) A[i] = sinf(B[i]);
; }
;

;
; This loop will be vectorized, although the scalar cost is lower than any of vector costs, but vectorization is explicitly forced in metadata.
;

define void @vectorized(ptr noalias nocapture %A, ptr noalias nocapture %B) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %B, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4, !llvm.access.group !11
  %call = tail call float @llvm.sin.f32(float %0)
  %arrayidx2 = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  store float %call, ptr %arrayidx2, align 4, !llvm.access.group !11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1000
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !llvm.loop !1

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

!1 = !{!1, !2, !{!"llvm.loop.parallel_accesses", !11}}
!2 = !{!"llvm.loop.vectorize.enable", i1 true}
!11 = distinct !{}

;
; This method will not be vectorized, as scalar cost is lower than any of vector costs.
;

define void @not_vectorized(ptr noalias nocapture %A, ptr noalias nocapture %B) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %B, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4, !llvm.access.group !13
  %call = tail call float @llvm.sin.f32(float %0)
  %arrayidx2 = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  store float %call, ptr %arrayidx2, align 4, !llvm.access.group !13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1000
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !llvm.loop !3

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

declare float @llvm.sin.f32(float) nounwind readnone

; Dummy metadata
!3 = !{!3, !{!"llvm.loop.parallel_accesses", !13}}
!13 = distinct !{}

