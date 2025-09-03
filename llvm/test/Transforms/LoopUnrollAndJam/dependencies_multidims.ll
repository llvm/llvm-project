; RUN: opt -da-disable-delinearization-checks -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-and-jam-count=4 < %s -S | FileCheck %s
; RUN: opt -da-disable-delinearization-checks -aa-pipeline=basic-aa -passes='loop-unroll-and-jam' -allow-unroll-and-jam -unroll-and-jam-count=4 < %s -S | FileCheck %s

; CHECK-LABEL: sub_sub_less
; CHECK: %j = phi
; CHECK-NOT: %j.1 = phi
;
; sub_sub_less should NOT be unroll-and-jammed due to a loop-carried dependency.
; Memory accesses:
;   - A[i][j] = 1        (write to current iteration)
;   - A[i+1][j-1] = add  (write to next i iteration, previous j iteration)
; The dependency: A[i+1][j-1] from iteration (i,j) may conflict with A[i'][j']
; from a later iteration when i'=i+1 and j'=j-1, creating a backward dependency
; in the j dimension that prevents safe unroll-and-jam.
define void @sub_sub_less(ptr noalias nocapture %A, i32 %N, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 1, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %arrayidx = getelementptr inbounds [100 x i32], ptr %A, i32 %i, i32 %j
  store i32 1, ptr %arrayidx, align 4
  %add72 = add nuw nsw i32 %i, 1
  %add73 = add nuw nsw i32 %j, -1
  %arrayidx8 = getelementptr inbounds [100 x i32], ptr %A, i32 %add72, i32 %add73
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
;   - A[i][j] = 1      (write to current iteration)
;   - A[i+1][j] = add  (write to next i iteration, same j iteration)
; No dependency conflict: A[i+1][j] from iteration (i,j) doesn't conflict with
; any A[i'][j'] from unrolled j iterations since j' values are different and
; i+1 from current doesn't overlap with i' from unrolled iterations.
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
  %arrayidx = getelementptr inbounds [100 x i32], ptr %A, i32 %i, i32 %j
  store i32 1, ptr %arrayidx, align 4
  %add72 = add nuw nsw i32 %i, 1
  %add73 = add nuw nsw i32 %j, 0
  %arrayidx8 = getelementptr inbounds [100 x i32], ptr %A, i32 %add72, i32 %add73
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
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
;
; sub_sub_more SHOULD be unroll-and-jammed (count=4) as it's safe.
; Memory accesses:
;   - A[i][j] = 1        (write to current iteration)
;   - A[i+1][j+1] = add  (write to next i iteration, next j iteration)
; No dependency conflict: A[i+1][j+1] from iteration (i,j) doesn't conflict with
; any A[i'][j'] from unrolled j iterations since the forward dependency pattern
; doesn't create overlapping accesses between unrolled iterations.
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
  %arrayidx = getelementptr inbounds [100 x i32], ptr %A, i32 %i, i32 %j
  store i32 1, ptr %arrayidx, align 4
  %add72 = add nuw nsw i32 %i, 1
  %add73 = add nuw nsw i32 %j, 1
  %arrayidx8 = getelementptr inbounds [100 x i32], ptr %A, i32 %add72, i32 %add73
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

; CHECK-LABEL: sub_sub_less_3d
; CHECK: %k = phi
; CHECK-NOT: %k.1 = phi
;
; sub_sub_less_3d should NOT be unroll-and-jammed due to a loop-carried dependency.
; Memory accesses:
;   - A3d[i][j][k] = 0     (write to current iteration)
;   - A3d[i+1][j][k-1] = 0 (write to next i iteration, previous k iteration)
; The dependency: A[i+1][j][k-1] from iteration (i,j,k) may conflict with
; A[i'][j'][k'] from a later iteration when i'=i+1 and k'=k-1, creating a
; backward dependency in the k dimension that prevents safe unroll-and-jam.
; This is a 3D version of the same pattern as sub_sub_less.
;
; for (long i = 0; i < 100; ++i)
;   for (long j = 0; j < 100; ++j)
;     for (long k = 1; k < 100; ++k) {
;       A[i][j][k] = 5;
;       A[i+1][j][k-1] = 10;
;     }

define void @sub_sub_less_3d(ptr noalias %A) {
entry:
  br label %for.i

for.i:
  %i = phi i32 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i ], [ %inc.j, %for.j.latch ]
  br label %for.k

for.k:
  %k = phi i32 [ 1, %for.j ], [ %inc.k, %for.k ]
  %arrayidx = getelementptr inbounds [100 x [100 x i32]], ptr %A, i32 %i, i32 %j, i32 %k
  store i32 5, ptr %arrayidx, align 4
  %add.i = add nsw i32 %i, 1
  %sub.k = add nsw i32 %k, -1
  %arrayidx2 = getelementptr inbounds [100 x [100 x i32]], ptr %A, i32 %add.i, i32 %j, i32 %sub.k
  store i32 10, ptr %arrayidx2, align 4
  %inc.k = add nsw i32 %k, 1
  %cmp.k = icmp slt i32 %inc.k, 100
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %inc.j = add nsw i32 %j, 1
  %cmp.j = icmp slt i32 %inc.j, 100
  br i1 %cmp.j, label %for.j, label %for.i.latch, !llvm.loop !1

for.i.latch:
  %inc.i = add nsw i32 %i, 1
  %cmp.i = icmp slt i32 %inc.i, 100
  br i1 %cmp.i, label %for.i, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: sub_sub_outer_scalar
; CHECK: %k = phi
; CHECK-NOT: %k.1 = phi
;
; sub_sub_outer_scalar should NOT be unroll-and-jammed due to a loop-carried dependency.
; Memory accesses:
;   - load from A[j][k]    (read from current j iteration)
;   - store to A[j-1][k]   (write to previous j iteration)
; The dependency: reading A[j][k] and writing A[j-1][k] creates a backward
; dependency in the j dimension. The test attempts to unroll-and-jam the j loop
; with the k loop being jammed. When this happens, iterations j, j+1, j+2, j+3
; would be unrolled and their k loops jammed together, but j+1's write to A[j][k]
; would conflict with j's read from A[j][k], violating sequential semantics.
define void @sub_sub_outer_scalar(ptr %A) {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i64 [ 1, %for.i ], [ %inc.j, %for.j.latch ]
  br label %for.k

for.k:
  %k = phi i64 [ 0, %for.j ], [ %inc.k, %for.k ]
  %arrayidx = getelementptr inbounds [100 x i32], ptr %A, i64 %j
  %arrayidx7 = getelementptr inbounds [100 x i32], ptr %arrayidx, i64 0, i64 %k
  %0 = load i32, ptr %arrayidx7, align 4
  %sub.j = sub nsw i64 %j, 1
  %arrayidx8 = getelementptr inbounds [100 x i32], ptr %A, i64 %sub.j
  %arrayidx9 = getelementptr inbounds [100 x i32], ptr %arrayidx8, i64 0, i64 %k
  store i32 %0, ptr %arrayidx9, align 4
  %inc.k = add nsw i64 %k, 1
  %cmp.k = icmp slt i64 %inc.k, 100
  br i1 %cmp.k, label %for.k, label %for.j.latch

for.j.latch:
  %inc.j = add nsw i64 %j, 1
  %cmp.j = icmp slt i64 %inc.j, 100
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %inc.i = add nsw i64 %i, 1
  %cmp.i = icmp slt i64 %inc.i, 100
  br i1 %cmp.i, label %for.i, label %for.end

for.end:
  ret void
}

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.unroll_and_jam.disable"}
