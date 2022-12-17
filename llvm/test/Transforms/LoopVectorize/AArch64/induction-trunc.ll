; RUN: opt < %s -force-vector-width=1 -force-vector-interleave=2 -passes=loop-vectorize -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; CHECK-LABEL: @non_primary_iv_trunc_free(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = mul i64 [[INDEX]], 5
; CHECK-NEXT:    [[INDUCTION:%.*]] = add i64 [[OFFSET_IDX]], 0
; CHECK-NEXT:    [[INDUCTION1:%.*]] = add i64 [[OFFSET_IDX]], 5
; CHECK-NEXT:    [[TMP4:%.*]] = trunc i64 [[INDUCTION]] to i32
; CHECK-NEXT:    [[TMP5:%.*]] = trunc i64 [[INDUCTION1]] to i32
; CHECK-NEXT:    [[GEP0:%.+]] = getelementptr inbounds i32, i32* %dst, i32 [[TMP4]]
; CHECK-NEXT:    [[GEP1:%.+]] = getelementptr inbounds i32, i32* %dst, i32 [[TMP5]]
; CHECK-NEXT:    store i32 0, i32* [[GEP0]], align 4
; CHECK-NEXT:    store i32 0, i32* [[GEP1]], align 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @non_primary_iv_trunc_free(i64 %n, i32* %dst) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = trunc i64 %i to i32
  %gep.dst = getelementptr inbounds i32, i32* %dst, i32 %tmp0
  store i32 0, i32* %gep.dst
  %i.next = add nuw nsw i64 %i, 5
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
