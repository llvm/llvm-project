; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -S | FileCheck %s

; Test case for PR45525. Checks that phi's with a single predecessor and a mask are supported.

define void @main(i1 %cond, ptr %arr) {
; CHECK-LABEL: @main(
; CHECK-NEXT:  bb.0:
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK:         br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK:         [[VEC_IND:%.*]] = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK:         [[TMP5:%.*]] = mul <4 x i32> [[VEC_IND]], splat (i32 3)
;
bb.0:
  br label %bb.1

bb.1:                                             ; preds = %bb.3, %bb.0
  %iv = phi i32 [ 0, %bb.0 ], [ %iv.next, %bb.3 ]
  br i1 %cond, label %bb.3, label %bb.2

bb.2:                                             ; preds = %bb.1
  %single.pred = phi i32 [ %iv, %bb.1 ]
  %mult = mul i32 %single.pred, 3
  br label %bb.3

bb.3:                                             ; preds = %bb.2, %bb.1
  %stored.value = phi i32 [ 7, %bb.1 ], [ %mult, %bb.2 ]
  %arrayidx = getelementptr inbounds i32, ptr %arr, i32 %iv
  store i32 %stored.value, ptr %arrayidx
  %iv.next = add i32 %iv, 1
  %continue = icmp ult i32 %iv.next, 32
  br i1 %continue, label %bb.1, label %bb.4

bb.4:                                             ; preds = %bb.3
  ret void
}
