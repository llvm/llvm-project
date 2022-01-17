; RUN: opt -S -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"

define void @foo(i64* %ptr, i32* %ptr.2) {
; CHECK-LABEL: @foo(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY:%.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 2, i64 3, i64 4, i64 5>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_TRUNC:%.+]] = phi <4 x i32> [ <i32 2, i32 3, i32 4, i32 5>, %vector.ph ], [ [[VEC_IND_TRUNC_NEXT:%.+]], %vector.body ]
; CHECK-NEXT:    = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = add i64 2, [[INDEX]]
; CHECK-NEXT:    [[TRUNC:%.+]] = trunc i64 [[OFFSET_IDX]] to i32
; CHECK-NEXT:    [[TMP7:%.*]] = add i32 [[TRUNC]], 0
; CHECK-NEXT:    [[TMP8:%.*]] = add i32 [[TRUNC]], 1
; CHECK-NEXT:    [[TMP9:%.*]] = add i32 [[TRUNC]], 2
; CHECK-NEXT:    [[TMP10:%.*]] = add i32 [[TRUNC]], 3
; CHECK-NEXT:    store i32 [[TMP7]], i32* %ptr.2, align 4
; CHECK-NEXT:    store i32 [[TMP8]], i32* %ptr.2, align 4
; CHECK-NEXT:    store i32 [[TMP9]], i32* %ptr.2, align 4
; CHECK-NEXT:    store i32 [[TMP10]], i32* %ptr.2, align 4
; CHECK:         store <4 x i64> [[VEC_IND]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <4 x i64> [[VEC_IND]], <i64 4, i64 4, i64 4, i64 4>
; CHECK-NEXT:    [[VEC_IND_TRUNC_NEXT]] = add <4 x i32> [[VEC_IND_TRUNC]], <i32 4, i32 4, i32 4, i32 4>
; CHECK-NEXT:    [[TMP13:%.*]] = icmp eq i64 [[INDEX_NEXT]], 80
; CHECK-NEXT:    br i1 [[TMP13]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]]
;
entry:
  br label %loop

loop:
  %can.iv = phi i64 [ 0, %entry ], [ %can.iv.next, %loop ]
  %0 = phi i64 [ 2, %entry ], [ %3, %loop ]
  %1 = and i64 %0, 4294967295
  %2 = trunc i64 %0 to i32
  store i32 %2, i32* %ptr.2
  %gep.ptr = getelementptr inbounds i64, i64* %ptr, i64 %can.iv
  store i64 %0, i64* %gep.ptr
  %3 = add nuw nsw i64 %1, 1
  %4 = icmp sgt i32 %2, 80
  %can.iv.next = add nuw nsw i64 %can.iv, 1
  br i1 %4, label %exit, label %loop

exit:
  ret void
}
