; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@array = common global [1024 x i32] zeroinitializer, align 16

define void @array_at_plus_one(i32 %n) {
; CHECK-LABEL: @array_at_plus_one(
; CHECK: %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK: [[VEC_IV_TRUNC:%.+]] = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %vector.ph ], [ [[VEC_IV_TRUNC_NEXT:%.+]], %vector.body ]
; CHECK: [[T1:%.+]] = add i64 %index, 0
; CHECK: [[T2:%.+]] = add nsw i64 [[T1]], 12
; CHECK-NEXT: [[GEP:%.+]] = getelementptr inbounds [1024 x i32], ptr @array, i64 0, i64 [[T2]]
; CHECK-NEXT: [[GEP0:%.+]] = getelementptr inbounds i32, ptr [[GEP]], i32 0
; CHECK-NEXT: store <4 x i32> [[VEC_IV_TRUNC]], ptr [[GEP0]]
; CHECK: [[VEC_IV_TRUNC_NEXT]] = add <4 x i32> [[VEC_IV_TRUNC]], <i32 4, i32 4, i32 4, i32 4>
; CHECK: ret void
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.plus.12 = add nsw i64 %iv, 12
  %gep = getelementptr inbounds [1024 x i32], ptr @array, i64 0, i64 %iv.plus.12
  %iv.trunc = trunc i64 %iv to i32
  store i32 %iv.trunc, ptr %gep, align 4
  %iv.next = add i64 %iv, 1
  %lftr.wideiv = trunc i64 %iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}
