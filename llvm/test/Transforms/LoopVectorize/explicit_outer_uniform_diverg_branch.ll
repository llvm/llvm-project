; RUN: opt < %s -passes=loop-vectorize -enable-vplan-native-path -debug-only=loop-vectorize --disable-output -S 2>&1 | FileCheck %s
; REQUIRES: asserts

; Verify that LV can handle explicit vectorization outer loops with uniform branches
; but bails out on outer loops with divergent branches.

; Root C/C++ source code for the first two test cases.
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

define void @uniform_branch(ptr nocapture %a, ptr nocapture readonly %b, i32 %N, i32 %M) {
entry:
  %cmp39 = icmp sgt i32 %N, 0
  br i1 %cmp39, label %outer.ph, label %for.end19

outer.ph:
  %cmp337 = icmp slt i32 %M, 1
  %0 = sext i32 %M to i64
  %N64 = zext i32 %N to i64
  %M64 = zext i32 %M to i64
  %cmp1 = icmp ne i32 %M, %N ; Uniform condition
  %brmerge = or i1 %cmp1, %cmp337 ; Uniform condition
  br label %outer.body

outer.body:
  %indvars.iv42 = phi i64 [ 0, %outer.ph ], [ %indvars.iv.next43, %outer.inc ]
  %1 = mul nsw i64 %indvars.iv42, %0
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %1
  %2 = load i32, ptr %arrayidx, align 4
  br i1 %brmerge, label %outer.inc, label %inner.ph ; Supported uniform branch

inner.ph:
  br label %inner.body

inner.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %inner.body ], [ 0, %inner.ph ]
  %3 = add nsw i64 %indvars.iv, %1
  %arrayidx7 = getelementptr inbounds i32, ptr %b, i64 %3
  %4 = load i32, ptr %arrayidx7, align 4
  %mul12 = mul nsw i32 %4, %4
  %arrayidx16 = getelementptr inbounds i32, ptr %a, i64 %3
  store i32 %mul12, ptr %arrayidx16, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %M64
  br i1 %exitcond, label %outer.inc, label %inner.body

outer.inc:
  %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
  %exitcond46 = icmp eq i64 %indvars.iv.next43, %N64
  br i1 %exitcond46, label %for.end19, label %outer.body, !llvm.loop !6

for.end19:
  ret void
}


; Case 2 (COND => B[i * M] == 0): Outer loop with divergent conditional branch.

; CHECK-LABEL: divergent_branch
; CHECK: LV: Not vectorizing: Outer loop contains divergent conditional branch.
; CHECK: LV: Not vectorizing: Unsupported outer loop.

define void @divergent_branch(ptr nocapture %a, ptr nocapture readonly %b, i32 %N, i32 %M) {
entry:
  %cmp39 = icmp sgt i32 %N, 0
  br i1 %cmp39, label %outer.ph, label %for.end19

outer.ph:
  %cmp337 = icmp slt i32 %M, 1
  %0 = sext i32 %M to i64
  %N64 = zext i32 %N to i64
  %M64 = zext i32 %M to i64
  br label %outer.body

outer.body:
  %indvars.iv42 = phi i64 [ 0, %outer.ph ], [ %indvars.iv.next43, %outer.inc ]
  %1 = mul nsw i64 %indvars.iv42, %0
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %1
  %2 = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp ne i32 %2, 0 ; Divergent condition
  %brmerge = or i1 %cmp1, %cmp337 ; Divergent condition
  br i1 %brmerge, label %outer.inc, label %inner.ph ; Unsupported divergent branch.

inner.ph:
  br label %inner.body

inner.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %inner.body ], [ 0, %inner.ph ]
  %3 = add nsw i64 %indvars.iv, %1
  %arrayidx7 = getelementptr inbounds i32, ptr %b, i64 %3
  %4 = load i32, ptr %arrayidx7, align 4
  %mul12 = mul nsw i32 %4, %4
  %arrayidx16 = getelementptr inbounds i32, ptr %a, i64 %3
  store i32 %mul12, ptr %arrayidx16, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %M64
  br i1 %exitcond, label %outer.inc, label %inner.body

outer.inc:
  %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
  %exitcond46 = icmp eq i64 %indvars.iv.next43, %N64
  br i1 %exitcond46, label %for.end19, label %outer.body, !llvm.loop !6

for.end19:
  ret void
}

; Case 3: Three-level loop nest with a triangular innermost loop.
;
;   #pragma clang loop vectorize(enable) vectorize_width(8)
;   for (size_t i = 0; i < M; ++i)
;     for (size_t j = 0; j < N; ++j)
;       for (size_t k = 0; k < j; ++k)
;         a[i * N * N + j * N + k] = b[i * N * N + j * N + k];
;
; The innermost loop latch condition depends on the middle loop IV, but is
; uniform with respect to the vectorized outer loop.

; CHECK-LABEL: uniform_triangular_inner_loop
; CHECK: LV: We can vectorize this outer loop!

define void @uniform_triangular_inner_loop(ptr nocapture %a, ptr nocapture readonly %b, i64 %M, i64 %N) {
entry:
  %cmp.m = icmp eq i64 %M, 0
  br i1 %cmp.m, label %exit, label %outer.ph

outer.ph:
  %cmp.n = icmp eq i64 %N, 0
  br label %outer.body

outer.body:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.inc ]
  br i1 %cmp.n, label %outer.inc, label %middle.body

middle.body:
  %j = phi i64 [ 0, %outer.body ], [ %j.next, %middle.inc ]
  %cmp.k = icmp eq i64 %j, 0
  br i1 %cmp.k, label %middle.inc, label %inner.body

inner.body:
  %k = phi i64 [ 0, %middle.body ], [ %k.next, %inner.body ]
  %mul.n.n = mul nuw i64 %N, %N
  %mul.i = mul nuw i64 %i, %mul.n.n
  %mul.j = mul nuw i64 %j, %N
  %add.j = add nuw i64 %mul.i, %mul.j
  %idx = add nuw i64 %add.j, %k
  %arrayidx.b = getelementptr inbounds i32, ptr %b, i64 %idx
  %0 = load i32, ptr %arrayidx.b, align 4
  %arrayidx.a = getelementptr inbounds i32, ptr %a, i64 %idx
  store i32 %0, ptr %arrayidx.a, align 4
  %k.next = add nuw i64 %k, 1
  %exitcond.k = icmp eq i64 %k.next, %j
  br i1 %exitcond.k, label %middle.inc, label %inner.body

middle.inc:
  %j.next = add nuw i64 %j, 1
  %exitcond.j = icmp eq i64 %j.next, %N
  br i1 %exitcond.j, label %outer.inc, label %middle.body

outer.inc:
  %i.next = add nuw i64 %i, 1
  %exitcond.i = icmp eq i64 %i.next, %M
  br i1 %exitcond.i, label %exit, label %outer.body, !llvm.loop !6

exit:
  ret void
}

!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.vectorize.width", i32 8}
!8 = !{!"llvm.loop.vectorize.enable", i1 true}
