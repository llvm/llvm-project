; RUN: opt -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-and-jam-count=4 < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='loop-unroll-and-jam' -allow-unroll-and-jam -unroll-and-jam-count=4 < %s -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK-LABEL: fore_aft_less
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
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
