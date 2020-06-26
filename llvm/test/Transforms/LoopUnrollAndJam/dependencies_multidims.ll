; RUN: opt -da-disable-delinearization-checks -basicaa -loop-unroll-and-jam -allow-unroll-and-jam -unroll-and-jam-count=4 < %s -S | FileCheck %s
; RUN: opt -da-disable-delinearization-checks -aa-pipeline=basic-aa -passes='loop-unroll-and-jam' -allow-unroll-and-jam -unroll-and-jam-count=4 < %s -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK-LABEL: sub_sub_less
; CHECK: %j = phi
; CHECK-NOT: %j.1 = phi
define void @sub_sub_less([100 x i32]* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* %A, i32 %i, i32 %j
  store i32 1, i32* %arrayidx, align 4
  %add72 = add nuw nsw i32 %i, 1
  %add73 = add nuw nsw i32 %j, -1
  %arrayidx8 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i32 %add72, i32 %add73
  store i32 %add, i32* %arrayidx8, align 4
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
define void @sub_sub_eq([100 x i32]* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* %A, i32 %i, i32 %j
  store i32 1, i32* %arrayidx, align 4
  %add72 = add nuw nsw i32 %i, 1
  %add73 = add nuw nsw i32 %j, 0
  %arrayidx8 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i32 %add72, i32 %add73
  store i32 %add, i32* %arrayidx8, align 4
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
define void @sub_sub_more([100 x i32]* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add, %for.inner ], [ 0, %for.outer ]
  %arrayidx5 = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add6 = add nuw nsw i32 %j, 1
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* %A, i32 %i, i32 %j
  store i32 1, i32* %arrayidx, align 4
  %add72 = add nuw nsw i32 %i, 1
  %add73 = add nuw nsw i32 %j, 1
  %arrayidx8 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i32 %add72, i32 %add73
  store i32 %add, i32* %arrayidx8, align 4
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

; for (long i = 0; i < 100; ++i)
;   for (long j = 0; j < 100; ++j)
;     for (long k = 0; k < 100; ++k) {
;       A[i][j][k] = 0;
;       A[i+1][j][k-1] = 0;
;     }

define void @sub_sub_less_3d([100 x [100 x i32]]* noalias %A) {
entry:
  br label %for.i

for.i:
  %i = phi i32 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i ], [ %inc.j, %for.j.latch ]
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.j ], [ %inc.k, %for.k ]
  %arrayidx = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* %A, i32 %i, i32 %j, i32 %k
  store i32 0, i32* %arrayidx, align 4
  %add.i = add nsw i32 %i, 1
  %sub.k = add nsw i32 %k, -1
  %arrayidx2 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* %A, i32 %add.i, i32 %j, i32 %sub.k
  store i32 0, i32* %arrayidx2, align 4
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

define void @sub_sub_outer_scalar([100 x i32]* %A) {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i ], [ %inc.j, %for.j.latch ]
  br label %for.k

for.k:
  %k = phi i64 [ 0, %for.j ], [ %inc.k, %for.k ]
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %j
  %arrayidx7 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx, i64 0, i64 %k
  %0 = load i32, i32* %arrayidx7, align 4
  %sub.j = sub nsw i64 %j, 1
  %arrayidx8 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %sub.j
  %arrayidx9 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx8, i64 0, i64 %k
  store i32 %0, i32* %arrayidx9, align 4
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
