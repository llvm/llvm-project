; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -mtriple=arm64-apple-darinw -S %s | FileCheck %s

; In the loop below, both the current and previous values of a first-order
; recurrence are stored in an interleave group.
define void @interleaved_store_first_order_recurrence(ptr noalias %src, ptr %dst) {
; CHECK-LABEL: @interleaved_store_first_order_recurrence(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i32> [ <i32 poison, i32 poison, i32 poison, i32 99>, %vector.ph ], [ [[BROADCAST_SPLAT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[SRC:%.*]], align 4
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[TMP1]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <4 x i32> [[VECTOR_RECUR]], <4 x i32> [[BROADCAST_SPLAT]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP3:%.*]] = mul nuw nsw i64 [[TMP0]], 3
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, ptr [[DST:%.*]], i64 [[TMP3]]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[TMP4]], i64 2
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, ptr [[TMP6]], i32 -2
; CHECK-NEXT:    [[TMP9:%.*]] = shufflevector <4 x i32> zeroinitializer, <4 x i32> [[TMP2]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:    [[TMP10:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLAT]], <4 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[TMP11:%.*]] = shufflevector <8 x i32> [[TMP9]], <8 x i32> [[TMP10]], <12 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
; CHECK-NEXT:    [[INTERLEAVED_VEC:%.*]] = shufflevector <12 x i32> [[TMP11]], <12 x i32> poison, <12 x i32> <i32 0, i32 4, i32 8, i32 1, i32 5, i32 9, i32 2, i32 6, i32 10, i32 3, i32 7, i32 11>
; CHECK-NEXT:    store <12 x i32> [[INTERLEAVED_VEC]], ptr [[TMP7]], align 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP12:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP12]], label %middle.block, label %vector.body
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %for = phi i32 [ 99, %entry ],[ %for.next, %loop ]
  %for.next = load i32, ptr %src, align 4
  %off = mul nuw nsw i64 %iv, 3
  %gep.1 = getelementptr inbounds i32, ptr %dst, i64 %off
  store i32 0, ptr %gep.1, align 4
  %gep.2 = getelementptr inbounds i32, ptr %gep.1, i64 1
  store i32 %for, ptr %gep.2, align 4
  %gep.3 = getelementptr inbounds i32, ptr %gep.1, i64 2
  store i32 %for.next, ptr %gep.3, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1000
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
