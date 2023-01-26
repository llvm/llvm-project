; RUN: opt < %s -passes=loop-vectorize -mtriple=ve-linux -S | FileCheck %s -check-prefix=VE
; RUN: opt < %s -passes=loop-vectorize -mtriple=x86_64-pc_linux -mcpu=core-avx2 -S | FileCheck %s -check-prefix=AVX

; Make sure LV does not trigger for VE on an appealing loop that vectorizes for x86 AVX.

; TODO: Remove this test once VE vector isel is deemed stable.

; VE-NOT: llvm.loop.isvectorized
; AVX: llvm.loop.isvectorized

define dso_local void @foo(ptr noalias nocapture %A, ptr noalias nocapture readonly %B, i32 signext %n) local_unnamed_addr {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %omp.inner.for.body.preheader, label %simd.if.end

omp.inner.for.body.preheader:                     ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %omp.inner.for.body

omp.inner.for.body:                               ; preds = %omp.inner.for.body.preheader, %omp.inner.for.body
  %indvars.iv = phi i64 [ 0, %omp.inner.for.body.preheader ], [ %indvars.iv.next, %omp.inner.for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4, !llvm.access.group !6
  %mul6 = mul nsw i32 %0, 3
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %mul6, ptr %arrayidx8, align 4, !llvm.access.group !6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %simd.if.end, label %omp.inner.for.body, !llvm.loop !7

simd.if.end:                                      ; preds = %omp.inner.for.body, %entry
  ret void
}

!6 = distinct !{}
!7 = distinct !{!7, !8, !9}
!8 = !{!"llvm.loop.parallel_accesses", !6}
!9 = !{!"llvm.loop.vectorize.enable", i1 true}
