; RUN: opt -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-and-jam-count=4 < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='loop-unroll-and-jam' -allow-unroll-and-jam -unroll-and-jam-count=4 < %s -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK-LABEL: fore_aft_less
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
;
; fore_aft_less SHOULD be unroll-and-jammed (count=4) as it's safe.
; Memory accesses:
;   - Fore block: A[i] = 1      (write in outer loop before inner)
;   - Aft block:  A[i-1] = sum  (write in outer loop after inner)
; No dependency conflict: The fore block write A[i] and aft block write A[i-1]
; access different array elements, so unrolling the outer loop and jamming the
; inner loop is safe. The backward dependency (i-1) doesn't create conflicts
; between different unrolled iterations.
define void @fore_aft_less(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 1, ptr %arrayidx, align 4
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %exitcond = icmp eq i32 %add6, %N
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add7 = add nuw nsw i32 %i, 1
  %add72 = add nuw nsw i32 %i, -1
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i32 %add72
  store i32 %add, ptr %arrayidx8, align 4
  %exitcond29 = icmp eq i32 %add7, %N
  br i1 %exitcond29, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: fore_aft_eq
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
;
; fore_aft_eq SHOULD be unroll-and-jammed (count=4) as it's safe.
; Memory accesses:
;   - Fore block: A[i] = 1    (write in outer loop before inner)
;   - Aft block:  A[i] = sum  (write in outer loop after inner)
; Dependency conflict: Both fore and aft blocks write to A[i], creating a
; write-after-write (WAW) dependency. However, this is safe for unroll-and-jam
; because the aft block write always happens after the fore block write in
; the same iteration, preserving the original execution order.
define void @fore_aft_eq(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 1, ptr %arrayidx, align 4
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %exitcond = icmp eq i32 %add6, %N
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add7 = add nuw nsw i32 %i, 1
  %add72 = add nuw nsw i32 %i, 0
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %add, ptr %arrayidx8, align 4
  %exitcond29 = icmp eq i32 %add7, %N
  br i1 %exitcond29, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: fore_aft_more
; CHECK: %j = phi
; CHECK-NOT: %j.1 = phi
;
; fore_aft_more should NOT be unroll-and-jammed due to a dependency violation.
; Memory accesses:
;   - Fore block: A[i] = 1      (write in outer loop before inner)
;   - Aft block:  A[i+1] = sum  (write in outer loop after inner)
; Dependency conflict: The fore block writes A[i] and aft block writes A[i+1].
; When unroll-and-jamming, iteration i's aft block writes A[i+1] which conflicts
; with iteration i+1's fore block write to A[i+1], creating a write-after-write
; race condition that violates the original sequential semantics.
define void @fore_aft_more(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 1, ptr %arrayidx, align 4
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %exitcond = icmp eq i32 %add6, %N
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add7 = add nuw nsw i32 %i, 1
  %add72 = add nuw nsw i32 %i, 1
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i32 %add72
  store i32 %add, ptr %arrayidx8, align 4
  %exitcond29 = icmp eq i32 %add7, %N
  br i1 %exitcond29, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: fore_sub_less
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
;
; fore_sub_less SHOULD be unroll-and-jammed (count=4) as it's safe.
; Memory accesses:
;   - Fore block: A[i] = 1      (write in outer loop before inner)
;   - Sub block:  A[i-1] = sum  (write inside inner loop)
; No dependency conflict: The fore block writes A[i] and sub block writes A[i-1].
; These access different array elements, so unroll-and-jam is safe. The backward
; dependency pattern doesn't create conflicts between unrolled iterations.
define void @fore_sub_less(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 1, ptr %arrayidx, align 4
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add72 = add nuw nsw i32 %i, -1
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i32 %add72
  store i32 %add, ptr %arrayidx8, align 4
  %add6 = add nuw nsw i32 %j, 1
  %exitcond = icmp eq i32 %add6, %N
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add7 = add nuw nsw i32 %i, 1
  %exitcond29 = icmp eq i32 %add7, %N
  br i1 %exitcond29, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: fore_sub_eq
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
;
; fore_sub_eq SHOULD be unroll-and-jammed (count=4) as it's safe.
; Memory accesses:
;   - Fore block: A[i] = 1    (write in outer loop before inner)
;   - Sub block:  A[i] = sum  (write inside inner loop)
; Dependency conflict: Both fore and sub blocks write to A[i], creating a
; write-after-write (WAW) dependency. However, this is safe for unroll-and-jam
; because the execution order is preserved: fore block executes first, then
; the entire inner loop (sub block) executes, maintaining the original semantics.
define void @fore_sub_eq(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 1, ptr %arrayidx, align 4
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add72 = add nuw nsw i32 %i, 0
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i32 %add72
  store i32 %add, ptr %arrayidx8, align 4
  %add6 = add nuw nsw i32 %j, 1
  %exitcond = icmp eq i32 %add6, %N
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add7 = add nuw nsw i32 %i, 1
  %exitcond29 = icmp eq i32 %add7, %N
  br i1 %exitcond29, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: fore_sub_more
; CHECK: %j = phi
; CHECK-NOT: %j.1 = phi
;
; fore_sub_more should NOT be unroll-and-jammed due to a dependency violation.
; Memory accesses:
;   - Fore block: A[i] = 1      (write in outer loop before inner)
;   - Sub block:  A[i+1] = sum  (write inside inner loop)
; Dependency conflict: The fore block writes A[i] and sub block writes A[i+1].
; When unroll-and-jamming, iteration i's fore block writes A[i] but iteration i's
; sub block writes A[i+1]. This conflicts with iteration i+1's fore block write
; to A[i+1], creating a write-after-write race condition.
define void @fore_sub_more(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 1, ptr %arrayidx, align 4
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add72 = add nuw nsw i32 %i, 1
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i32 %add72
  store i32 %add, ptr %arrayidx8, align 4
  %add6 = add nuw nsw i32 %j, 1
  %exitcond = icmp eq i32 %add6, %N
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add7 = add nuw nsw i32 %i, 1
  %exitcond29 = icmp eq i32 %add7, %N
  br i1 %exitcond29, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: sub_aft_less
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
;
; sub_aft_less SHOULD be unroll-and-jammed (count=4) as it's safe.
; Memory accesses:
;   - Sub block: A[i] = 1      (write inside inner loop)
;   - Aft block: A[i-1] = sum  (write in outer loop after inner)
; No dependency conflict: The sub block writes A[i] and aft block writes A[i-1].
; These access different array elements, so unroll-and-jam is safe. The backward
; dependency pattern doesn't create conflicts between unrolled iterations.
define void @sub_aft_less(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 1, ptr %arrayidx, align 4
  %exitcond = icmp eq i32 %add6, %N
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add7 = add nuw nsw i32 %i, 1
  %add72 = add nuw nsw i32 %i, -1
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i32 %add72
  store i32 %add, ptr %arrayidx8, align 4
  %exitcond29 = icmp eq i32 %add7, %N
  br i1 %exitcond29, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: sub_aft_eq
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
;
; sub_aft_eq SHOULD be unroll-and-jammed (count=4) as it's safe.
; Memory accesses:
;   - Sub block: A[i] = 1    (write inside inner loop)
;   - Aft block: A[i] = sum  (write in outer loop after inner)
; Dependency conflict: Both sub and aft blocks write to A[i], creating a
; write-after-write (WAW) dependency. However, this is safe for unroll-and-jam
; because the execution order is preserved: the entire inner loop (sub block)
; executes first, then the aft block executes, maintaining original semantics.
define void @sub_aft_eq(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 1, ptr %arrayidx, align 4
  %exitcond = icmp eq i32 %add6, %N
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add7 = add nuw nsw i32 %i, 1
  %add72 = add nuw nsw i32 %i, 0
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %add, ptr %arrayidx8, align 4
  %exitcond29 = icmp eq i32 %add7, %N
  br i1 %exitcond29, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: sub_aft_more
; CHECK: %j = phi
; CHECK-NOT: %j.1 = phi
;
; sub_aft_more should NOT be unroll-and-jammed due to a dependency violation.
; Memory accesses:
;   - Sub block: A[i] = 1      (write inside inner loop)
;   - Aft block: A[i+1] = sum  (write in outer loop after inner)
; Dependency conflict: The sub block writes A[i] and aft block writes A[i+1].
; When unroll-and-jamming, iteration i's aft block writes A[i+1] which conflicts
; with iteration i+1's sub block write to A[i+1], creating a write-after-write
; race condition that violates the original sequential semantics.
define void @sub_aft_more(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 1, ptr %arrayidx, align 4
  %exitcond = icmp eq i32 %add6, %N
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add7 = add nuw nsw i32 %i, 1
  %add72 = add nuw nsw i32 %i, 1
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i32 %add72
  store i32 %add, ptr %arrayidx8, align 4
  %exitcond29 = icmp eq i32 %add7, %N
  br i1 %exitcond29, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: sub_sub_less
; CHECK: %j = phi
; CHECK-NOT: %j.1 = phi
;
; sub_sub_less should NOT be unroll-and-jammed due to a dependency violation.
; Memory accesses:
;   - Sub block: A[i] = 1      (write inside inner loop)
;   - Sub block: A[i-1] = sum  (write inside inner loop)
; Dependency conflict: Both writes are in the sub block (inner loop), accessing
; A[i] and A[i-1]. When unroll-and-jamming, the inner loop is jammed, meaning
; iterations of the inner loop from different outer iterations execute together.
; This creates a backward dependency that can cause race conditions.
define void @sub_sub_less(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 1, ptr %arrayidx, align 4
  %add72 = add nuw nsw i32 %i, -1
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i32 %add72
  store i32 %add, ptr %arrayidx8, align 4
  %exitcond = icmp eq i32 %add6, %N
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add7 = add nuw nsw i32 %i, 1
  %exitcond29 = icmp eq i32 %add7, %N
  br i1 %exitcond29, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: sub_sub_eq
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
;
; sub_sub_eq SHOULD be unroll-and-jammed (count=4) as it's safe.
; Memory accesses:
;   - Sub block: A[i] = 1    (write inside inner loop)
;   - Sub block: A[i] = sum  (write inside inner loop)
; Dependency conflict: Both writes are to A[i] within the sub block, creating a
; write-after-write (WAW) dependency. However, this is safe for unroll-and-jam
; because both writes are in the same basic block and maintain their relative
; order: A[i] = 1 always executes before A[i] = sum in each iteration.
define void @sub_sub_eq(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 1, ptr %arrayidx, align 4
  %add72 = add nuw nsw i32 %i, 0
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i32 %add72
  store i32 %add, ptr %arrayidx8, align 4
  %exitcond = icmp eq i32 %add6, %N
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add7 = add nuw nsw i32 %i, 1
  %exitcond29 = icmp eq i32 %add7, %N
  br i1 %exitcond29, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: sub_sub_more
; CHECK: %j = phi
; CHECK-NOT: %j.1 = phi
;
; sub_sub_more should NOT be unroll-and-jammed due to a dependency violation.
; Memory accesses:
;   - Sub block: A[i] = 1      (write inside inner loop)
;   - Sub block: A[i+1] = sum  (write inside inner loop)
; Dependency conflict: Both writes are in the sub block, accessing A[i] and A[i+1].
; When unroll-and-jamming, iteration i's sub block writes A[i+1] which conflicts
; with iteration i+1's sub block write to A[i+1]. This creates a forward
; dependency that causes write-after-write race conditions.
define void @sub_sub_more(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 1, ptr %arrayidx, align 4
  %add72 = add nuw nsw i32 %i, 1
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i32 %add72
  store i32 %add, ptr %arrayidx8, align 4
  %exitcond = icmp eq i32 %add6, %N
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add7 = add nuw nsw i32 %i, 1
  %exitcond29 = icmp eq i32 %add7, %N
  br i1 %exitcond29, label %cleanup, label %for.outer

cleanup:
  ret void
}
