; RUN: opt -S -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"

define void @foo() {
; CHECK-LABEL: @foo(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY:%.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 2, i64 3, i64 4, i64 5>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = add i64 2, [[INDEX]]
; CHECK-NEXT:    [[TMP7:%.*]] = add i64 [[OFFSET_IDX]], 0
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[OFFSET_IDX]], 1
; CHECK-NEXT:    [[TMP9:%.*]] = add i64 [[OFFSET_IDX]], 2
; CHECK-NEXT:    [[TMP10:%.*]] = add i64 [[OFFSET_IDX]], 3
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 4
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <4 x i64> [[VEC_IND]], <i64 4, i64 4, i64 4, i64 4>
; CHECK-NEXT:    [[TMP13:%.*]] = icmp eq i64 [[INDEX_NEXT]], 80
; CHECK-NEXT:    br i1 [[TMP13]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop !0
;
entry:
  br label %loop

loop:
  %0 = phi i64 [ 2, %entry ], [ %3, %loop ]
  %1 = and i64 %0, 4294967295
  %2 = trunc i64 %0 to i32
  %3 = add nuw nsw i64 %1, 1
  %4 = icmp sgt i32 %2, 80
  br i1 %4, label %exit, label %loop

exit:
  ret void
}
