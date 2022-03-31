; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @add_ints(i32* nocapture %A, i32* nocapture %B, i32* nocapture %C) {
; CHECK-LABEL: @add_ints(
; CHECK-LABEL: vector.memcheck:
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr i32, i32* [[A:%.*]], i64 200
; CHECK-NEXT:    [[SCEVGEP4:%.*]] = getelementptr i32, i32* [[B:%.*]], i64 200
; CHECK-NEXT:    [[SCEVGEP7:%.*]] = getelementptr i32, i32* [[C:%.*]], i64 200
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ugt i32* [[SCEVGEP4]], [[A]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ugt i32* [[SCEVGEP]], [[B]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    [[BOUND09:%.*]] = icmp ugt i32* [[SCEVGEP7]], [[A]]
; CHECK-NEXT:    [[BOUND110:%.*]] = icmp ugt i32* [[SCEVGEP]], [[C]]
; CHECK-NEXT:    [[FOUND_CONFLICT11:%.*]] = and i1 [[BOUND09]], [[BOUND110]]
; CHECK-NEXT:    [[CONFLICT_RDX:%.*]] = or i1 [[FOUND_CONFLICT]], [[FOUND_CONFLICT11]]
; CHECK-NEXT:    br i1 [[CONFLICT_RDX]], label %scalar.ph, label %vector.ph
; CHECK:       vector.ph:
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %add, i32* %arrayidx4, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 200
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
