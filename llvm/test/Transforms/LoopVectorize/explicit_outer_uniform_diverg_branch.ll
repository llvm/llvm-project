; RUN: opt < %s -passes=loop-vectorize -enable-vplan-native-path -debug-only=loop-vectorize -S 2>&1 | FileCheck %s
; REQUIRES: asserts

; Verify that LV can handle explicit vectorization outer loops with uniform branches
; but bails out on outer loops with divergent branches.

; Root C/C++ source code for the test cases
; void foo(int *a, int *b, int N, int M)
; {
;   int i, j;
; #pragma clang loop vectorize(enable) vectorize_width(8)
;   for (i = 0; i < N; i++) {
;     // Tested conditional branch. COND will be replaced per test.
;     if (COND)
;       for (j = 0; j < M; j++) {
;         a[i*M+j] = bptr b[i*M+j];
;       }
;   }
; }

; Case 1 (COND => M == N): Outer loop with uniform conditional branch.

; CHECK-LABEL: uniform_branch
; CHECK: LV: We can vectorize this outer loop!

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @uniform_branch(ptr nocapture %a, ptr nocapture readonly %b, i32 %N, i32 %M) local_unnamed_addr {
entry:
  %cmp39 = icmp sgt i32 %N, 0
  br i1 %cmp39, label %outer.ph, label %for.end19

outer.ph:                                   ; preds = %entry
  %cmp337 = icmp slt i32 %M, 1
  %0 = sext i32 %M to i64
  %N64 = zext i32 %N to i64
  %M64 = zext i32 %M to i64
  %cmp1 = icmp ne i32 %M, %N ; Uniform condition
  %brmerge = or i1 %cmp1, %cmp337 ; Uniform condition
  br label %outer.body

outer.body:                                 ; preds = %outer.inc, %outer.ph
  %indvars.iv42 = phi i64 [ 0, %outer.ph ], [ %indvars.iv.next43, %outer.inc ]
  %1 = mul nsw i64 %indvars.iv42, %0
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %1
  %2 = load i32, ptr %arrayidx, align 4, !tbaa !2
  br i1 %brmerge, label %outer.inc, label %inner.ph ; Supported uniform branch

inner.ph:                                   ; preds = %outer.body
  br label %inner.body

inner.body:                                 ; preds = %inner.ph, %inner.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %inner.body ], [ 0, %inner.ph ]
  %3 = add nsw i64 %indvars.iv, %1
  %arrayidx7 = getelementptr inbounds i32, ptr %b, i64 %3
  %4 = load i32, ptr %arrayidx7, align 4, !tbaa !2
  %mul12 = mul nsw i32 %4, %4
  %arrayidx16 = getelementptr inbounds i32, ptr %a, i64 %3
  store i32 %mul12, ptr %arrayidx16, align 4, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %M64
  br i1 %exitcond, label %outer.inc, label %inner.body

outer.inc:                                  ; preds = %inner.body, %outer.body
  %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
  %exitcond46 = icmp eq i64 %indvars.iv.next43, %N64
  br i1 %exitcond46, label %for.end19, label %outer.body, !llvm.loop !6

for.end19:                                  ; preds = %outer.inc, %entry
  ret void
}


; Case 2 (COND => B[i * M] == 0): Outer loop with divergent conditional branch.

; CHECK-LABEL: divergent_branch
; CHECK: Unsupported conditional branch.
; CHECK: LV: Not vectorizing: Unsupported outer loop.

define void @divergent_branch(ptr nocapture %a, ptr nocapture readonly %b, i32 %N, i32 %M) local_unnamed_addr {
entry:
  %cmp39 = icmp sgt i32 %N, 0
  br i1 %cmp39, label %outer.ph, label %for.end19

outer.ph:                                   ; preds = %entry
  %cmp337 = icmp slt i32 %M, 1
  %0 = sext i32 %M to i64
  %N64 = zext i32 %N to i64
  %M64 = zext i32 %M to i64
  br label %outer.body

outer.body:                                 ; preds = %outer.inc, %outer.ph
  %indvars.iv42 = phi i64 [ 0, %outer.ph ], [ %indvars.iv.next43, %outer.inc ]
  %1 = mul nsw i64 %indvars.iv42, %0
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %1
  %2 = load i32, ptr %arrayidx, align 4, !tbaa !2
  %cmp1 = icmp ne i32 %2, 0 ; Divergent condition
  %brmerge = or i1 %cmp1, %cmp337 ; Divergent condition
  br i1 %brmerge, label %outer.inc, label %inner.ph ; Unsupported divergent branch.

inner.ph:                                   ; preds = %outer.body
  br label %inner.body

inner.body:                                 ; preds = %inner.ph, %inner.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %inner.body ], [ 0, %inner.ph ]
  %3 = add nsw i64 %indvars.iv, %1
  %arrayidx7 = getelementptr inbounds i32, ptr %b, i64 %3
  %4 = load i32, ptr %arrayidx7, align 4, !tbaa !2
  %mul12 = mul nsw i32 %4, %4
  %arrayidx16 = getelementptr inbounds i32, ptr %a, i64 %3
  store i32 %mul12, ptr %arrayidx16, align 4, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %M64
  br i1 %exitcond, label %outer.inc, label %inner.body

outer.inc:                                  ; preds = %inner.body, %outer.body
  %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
  %exitcond46 = icmp eq i64 %indvars.iv.next43, %N64
  br i1 %exitcond46, label %for.end19, label %outer.body, !llvm.loop !6

for.end19:                                  ; preds = %outer.inc, %entry
  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.vectorize.width", i32 8}
!8 = !{!"llvm.loop.vectorize.enable", i1 true}
