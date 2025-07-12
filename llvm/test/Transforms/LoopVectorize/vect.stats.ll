; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=4 -force-vector-width=4 -debug-only=loop-vectorize -enable-early-exit-vectorization --disable-output -stats -S 2>&1 | FileCheck %s
; REQUIRES: asserts

; We have 3 loops, two of them are vectorizable (with one being early-exit
; vectorized) and the third one is not.

; CHECK: 3 loop-vectorize               - Number of loops analyzed for vectorization
; CHECK: 1 loop-vectorize               - Number of early exit loops vectorized
; CHECK: 2 loop-vectorize               - Number of loops vectorized

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @vectorized(ptr nocapture %a, i64 %size) {
entry:
  %cmp1 = icmp sle i64 %size, 0
  %cmp21 = icmp sgt i64 0, %size
  %or.cond = or i1 %cmp1, %cmp21
  br i1 %or.cond, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv2 = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv2
  %0 = load float, ptr %arrayidx, align 4
  %mul = fmul float %0, %0
  store float %mul, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv2, 1
  %cmp2 = icmp sgt i64 %indvars.iv.next, %size
  br i1 %cmp2, label %for.end, label %for.body

for.end:                                          ; preds = %entry, %for.body
  ret void
}

define i32 @early_exit_vectorized(i64 %end) {
entry:
  %p1 = alloca [1024 x i32]
  %p2 = alloca [1024 x i32]
  call void @init_mem(ptr %p1, i64 1024)
  call void @init_mem(ptr %p2, i64 1024)
  %end.clamped = and i64 %end, 1023
  br label %for.body

for.body:
  %ind = phi i64 [ %ind.next, %for.inc ], [ 0, %entry ]
  %arrayidx1 = getelementptr inbounds i32, ptr %p1, i64 %ind
  %0 = load i32, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %p2, i64 %ind
  %1 = load i32, ptr %arrayidx2, align 4
  %cmp.early = icmp eq i32 %0, %1
  br i1 %cmp.early, label %found, label %for.inc

for.inc:
  %ind.next = add i64 %ind, 1
  %cmp = icmp ult i64 %ind.next, %end.clamped
  br i1 %cmp, label %for.body, label %exit

found:
  ret i32 1

exit:
  ret i32 0
}

define void @not_vectorized(ptr nocapture %a, i64 %size) {
entry:
  %cmp1 = icmp sle i64 %size, 0
  %cmp21 = icmp sgt i64 0, %size
  %or.cond = or i1 %cmp1, %cmp21
  br i1 %or.cond, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv2 = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %0 = add nsw i64 %indvars.iv2, -5
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %0
  %1 = load float, ptr %arrayidx, align 4
  %2 = add nsw i64 %indvars.iv2, 2
  %arrayidx2 = getelementptr inbounds float, ptr %a, i64 %2
  %3 = load float, ptr %arrayidx2, align 4
  %mul = fmul float %1, %3
  %arrayidx4 = getelementptr inbounds float, ptr %a, i64 %indvars.iv2
  store float %mul, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv2, 1
  %cmp2 = icmp sgt i64 %indvars.iv.next, %size
  br i1 %cmp2, label %for.end, label %for.body

for.end:                                          ; preds = %entry, %for.body
  ret void
}

declare void @init_mem(ptr, i64);
