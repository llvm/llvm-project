; RUN: opt -loop-unroll -unroll-allow-partial -S %s -verify-loop-info -verify-dom-info -verify-loop-lcssa | FileCheck %s

@table = internal unnamed_addr global [344 x i32] zeroinitializer, align 16

define i32 @test_partial_unroll_with_breakout_at_iter0() {
; CHECK-LABEL: define i32 @test_partial_unroll_with_breakout_at_iter0() {
; CHECK-LABEL: entry:
; CHECK-NEXT:    br label %for.header

; CHECK-LABEL: for.header:                                       ; preds = %for.latch.3, %entry
; CHECK-NEXT:    %red = phi i32 [ 0, %entry ], [ %red.next.3, %for.latch.3 ]
; CHECK-NEXT:    %iv = phi i64 [ 0, %entry ], [ %iv.next.3, %for.latch.3 ]
; CHECK-NEXT:    %red.next = add nuw nsw i32 10, %red
; CHECK-NEXT:    %iv.next = add nuw nsw i64 %iv, 2
; CHECK-NEXT:    %ptr = getelementptr inbounds [344 x i32], [344 x i32]* @table, i64 0, i64 %iv.next
; CHECK-NEXT:    store i32 %red.next, i32* %ptr, align 4
; CHECK-NEXT:    br label %for.latch

; CHECK-LABEL: for.latch:                                        ; preds = %for.header
; CHECK-NEXT:    %red.next.1 = add nuw nsw i32 10, %red.next
; CHECK-NEXT:    %iv.next.1 = add nuw nsw i64 %iv.next, 2
; CHECK-NEXT:    %ptr.1 = getelementptr inbounds [344 x i32], [344 x i32]* @table, i64 0, i64 %iv.next.1
; CHECK-NEXT:    store i32 %red.next.1, i32* %ptr.1, align 4
; CHECK-NEXT:    br label %for.latch.1

; CHECK-LABEL: exit:                                             ; preds = %for.latch.2
; CHECK-NEXT:    ret i32 0

; CHECK-LABEL: for.latch.1:                                      ; preds = %for.latch
; CHECK-NEXT:    %red.next.2 = add nuw nsw i32 10, %red.next.1
; CHECK-NEXT:    %iv.next.2 = add nuw nsw i64 %iv.next.1, 2
; CHECK-NEXT:    %ptr.2 = getelementptr inbounds [344 x i32], [344 x i32]* @table, i64 0, i64 %iv.next.2
; CHECK-NEXT:    store i32 %red.next.2, i32* %ptr.2, align 4
; CHECK-NEXT:    br label %for.latch.2

; CHECK-LABEL: for.latch.2:                                      ; preds = %for.latch.1
; CHECK-NEXT:    %red.next.3 = add nuw nsw i32 10, %red.next.2
; CHECK-NEXT:    %iv.next.3 = add nuw nsw i64 %iv.next.2, 2
; CHECK-NEXT:    %ptr.3 = getelementptr inbounds [344 x i32], [344 x i32]* @table, i64 0, i64 %iv.next.3
; CHECK-NEXT:    store i32 %red.next.3, i32* %ptr.3, align 4
; CHECK-NEXT:    %exitcond.1.i.3 = icmp eq i64 %iv.next.3, 344
; CHECK-NEXT:    br i1 %exitcond.1.i.3, label %exit, label %for.latch.3

; CHECK-LABEL: for.latch.3:                                      ; preds = %for.latch.2
; CHECK-NEXT:    br label %for.header
;
entry:
  br label %for.header

for.header:                                     ; preds = %for.body28.i.for.body28.i_crit_edge, %for.body.i
  %red = phi i32 [ 0, %entry ], [ %red.next, %for.latch ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.latch ]
  %red.next = add i32 10, %red
  %iv.next = add nuw nsw i64 %iv, 2
  %ptr = getelementptr inbounds [344 x i32], [344 x i32]* @table, i64 0, i64 %iv.next
  store i32 %red.next, i32* %ptr, align 4
  %exitcond.1.i = icmp eq i64 %iv.next, 344
  br i1 %exitcond.1.i, label %exit, label %for.latch

for.latch:              ; preds = %for.header
  br label %for.header

exit:
  ret i32 0
}
