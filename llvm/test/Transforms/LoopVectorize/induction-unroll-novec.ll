; RUN: opt -passes=loop-vectorize -force-vector-interleave=2 -force-vector-width=1 -S %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"


; Test for PR54427.
define void @test_nonconst_start_and_step(ptr %dst, i32 %start, i32 %step, i64 %N) {
; CHECK-LABEL: @test_nonconst_start_and_step(
; CHECK:         [[NEG_STEP:%.+]] = sub i32 0, %step
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[INDUCTION3:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[INDUCTION4:%.*]] = add i64 [[INDEX]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = trunc i64 [[INDEX]] to i32
; CHECK-NEXT:    [[TMP3:%.*]] = mul i32 [[TMP2]], [[NEG_STEP]]
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = add i32 %start, [[TMP3]]
; CHECK-NEXT:    [[TMP4:%.*]] = mul i32 0, [[NEG_STEP]]
; CHECK-NEXT:    [[INDUCTION:%.*]] = add i32 [[OFFSET_IDX]], [[TMP4]]
; CHECK-NEXT:    [[TMP5:%.*]] = mul i32 1, [[NEG_STEP]]
; CHECK-NEXT:    [[INDUCTION2:%.*]] = add i32 [[OFFSET_IDX]], [[TMP5]]
; CHECK-NEXT:    [[TMP6:%.*]] = sub nsw i32 [[INDUCTION]], %step
; CHECK-NEXT:    [[TMP7:%.*]] = sub nsw i32 [[INDUCTION2]], %step
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, ptr [[DST:%.*]], i64 [[INDUCTION3]]
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i32, ptr [[DST]], i64 [[INDUCTION4]]
; CHECK-NEXT:    store i32 [[TMP6]], ptr [[TMP8]], align 2
; CHECK-NEXT:    store i32 [[TMP7]], ptr [[TMP9]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK-NEXT:    [[TMP10:%.*]] = icmp eq i64 [[INDEX_NEXT]]
; CHECK-NEXT:    br i1 [[TMP10]], label %middle.block, label %vector.body
;
entry:
  br label %loop

loop:
  %primary.iv = phi i64 [ 0, %entry ], [ %primary.iv.next, %loop ]
  %iv.down = phi i32 [ %start, %entry ], [ %iv.down.next, %loop ]
  %iv.down.next = sub nsw i32 %iv.down, %step
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 %primary.iv
  store i32 %iv.down.next, ptr %gep.dst, align 2
  %primary.iv.next = add nuw nsw i64 %primary.iv, 1
  %exitcond = icmp eq i64 %primary.iv.next, %N
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}
