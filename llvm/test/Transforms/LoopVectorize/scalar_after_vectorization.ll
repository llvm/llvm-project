; RUN: opt < %s -force-vector-width=4 -force-vector-interleave=2 -loop-vectorize -instcombine -S | FileCheck %s
; RUN: opt < %s -force-vector-width=4 -force-vector-interleave=2 -loop-vectorize -S | FileCheck %s --check-prefix=NO-IC

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: @scalar_after_vectorization_0
;
; CHECK: vector.body:
; CHECK:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:   %offset.idx = or i64 %index, 1
; CHECK:   %[[T2:.+]] = add nuw nsw i64 %offset.idx, %tmp0
; CHECK:   %[[T3:.+]] = sub nsw i64 %[[T2]], %x
; CHECK:   %[[T4:.+]] = getelementptr inbounds i32, i32* %a, i64 %[[T3]]
; CHECK:   %[[T5:.+]] = bitcast i32* %[[T4]] to <4 x i32>*
; CHECK:   load <4 x i32>, <4 x i32>* %[[T5]], align 4
; CHECK:   %[[T6:.+]] = getelementptr inbounds i32, i32* %[[T4]], i64 4
; CHECK:   %[[T7:.+]] = bitcast i32* %[[T6]] to <4 x i32>*
; CHECK:   load <4 x i32>, <4 x i32>* %[[T7]], align 4
; CHECK:   br {{.*}}, label %middle.block, label %vector.body
;
; NO-IC-LABEL: @scalar_after_vectorization_0
;
; NO-IC: vector.body:
; NO-IC:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; NO-IC:   %offset.idx = add i64 1, %index
; NO-IC:   %[[T2:.+]] = add i64 %offset.idx, 0
; NO-IC:   %[[T3:.+]] = add i64 %offset.idx, 4
; NO-IC:   %[[T4:.+]] = add nuw nsw i64 %[[T2]], %tmp0
; NO-IC:   %[[T5:.+]] = add nuw nsw i64 %[[T3]], %tmp0
; NO-IC:   %[[T6:.+]] = sub nsw i64 %[[T4]], %x
; NO-IC:   %[[T7:.+]] = sub nsw i64 %[[T5]], %x
; NO-IC:   %[[T8:.+]] = getelementptr inbounds i32, i32* %a, i64 %[[T6]]
; NO-IC:   %[[T9:.+]] = getelementptr inbounds i32, i32* %a, i64 %[[T7]]
; NO-IC:   %[[T10:.+]] = getelementptr inbounds i32, i32* %[[T8]], i32 0
; NO-IC:   %[[T11:.+]] = bitcast i32* %[[T10]] to <4 x i32>*
; NO-IC:   load <4 x i32>, <4 x i32>* %[[T11]], align 4
; NO-IC:   %[[T12:.+]] = getelementptr inbounds i32, i32* %[[T8]], i32 4
; NO-IC:   %[[T13:.+]] = bitcast i32* %[[T12]] to <4 x i32>*
; NO-IC:   load <4 x i32>, <4 x i32>* %[[T13]], align 4
; NO-IC:   br {{.*}}, label %middle.block, label %vector.body
;
define void @scalar_after_vectorization_0(i32* noalias %a, i32* noalias %b, i64 %x, i64 %y) {

outer.ph:
  br label %outer.body

outer.body:
  %i = phi i64 [ 1, %outer.ph ], [ %i.next, %inner.end ]
  %tmp0 = mul nuw nsw i64 %i, %x
  br label %inner.ph

inner.ph:
  br label %inner.body

inner.body:
  %j = phi i64 [ 1, %inner.ph ], [ %j.next, %inner.body ]
  %tmp1 = add nuw nsw i64 %j, %tmp0
  %tmp2 = sub nsw i64 %tmp1, %x
  %tmp3 = getelementptr inbounds i32, i32* %a, i64 %tmp2
  %tmp4 = load i32, i32* %tmp3, align 4
  %tmp5 = getelementptr inbounds i32, i32* %b, i64 %tmp1
  store i32 %tmp4, i32* %tmp5, align 4
  %j.next = add i64 %j, 1
  %cond.j = icmp slt i64 %j.next, %y
  br i1 %cond.j, label %inner.body, label %inner.end

inner.end:
  %i.next = add i64 %i, 1
  %cond.i = icmp slt i64 %i.next, %y
  br i1 %cond.i, label %outer.body, label %outer.end

outer.end:
  ret void
}
