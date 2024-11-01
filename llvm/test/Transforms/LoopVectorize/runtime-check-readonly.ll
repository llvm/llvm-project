; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @add_ints(i32* nocapture %A, i32* nocapture %B, i32* nocapture %C) {
; CHECK-LABEL: @add_ints(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_MEMCHECK:%.*]]
; CHECK-LABEL: vector.memcheck:
; CHECK-NEXT:    [[A1:%.*]] = ptrtoint i32* [[A:%.*]] to i64
; CHECK-NEXT:    [[B2:%.*]] = ptrtoint i32* [[B:%.*]] to i64
; CHECK-NEXT:    [[C3:%.*]] = ptrtoint i32* [[C:%.*]] to i64
; CHECK-NEXT:    [[TMP0:%.*]] = sub i64 [[A1]], [[B2]]
; CHECK-NEXT:    [[DIFF_CHECK:%.*]] = icmp ult i64 [[TMP0]], 16
; CHECK-NEXT:    [[TMP1:%.*]] = sub i64 [[A1]], [[C3]]
; CHECK-NEXT:    [[DIFF_CHECK4:%.*]] = icmp ult i64 [[TMP1]], 16
; CHECK-NEXT:    [[CONFLICT_RDX:%.*]] = or i1 [[DIFF_CHECK]], [[DIFF_CHECK4]]
; CHECK-NEXT:    br i1 [[CONFLICT_RDX]], label [[SCALAR_PH]], label [[VECTOR_PH:%.*]]
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
