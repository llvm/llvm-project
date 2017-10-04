; RUN: opt -irce -S < %s 2>&1 | FileCheck %s

; This test demonstrates a miscompile: the outer loop's IV iterates in range of
; [2, 400) and the range check is done against value 331. Due to a bug in range
; intersection IRCE manages to eliminate the range check without inserting a
; postloop, which is incorrect. So far IRCE is prohibited for this case.

define void @test_01() {

; CHECK-LABEL: test_01
; CHECK-NOT:   br i1 true

entry:
  br label %loop_header

loop_header:                            ; preds = %loop_latch, %entry
  %iv = phi i32 [ 2, %entry ], [ %iv_next, %loop_latch ]
  %iv.prev = phi i32 [ 1, %entry ], [ %iv, %loop_latch ]
  %tmp2 = icmp sgt i32 %iv.prev, -1
  br i1 %tmp2, label %loop_header.split.us, label %exit

loop_header.split.us:                   ; preds = %loop_header
  br label %inner_loop

inner_loop:                                   ; preds = %inner_loop, %loop_header.split.us
  %inner_iv = phi i32 [ 1, %loop_header.split.us ], [ %inner_iv_next, %inner_loop ]
  %inner_iv_next = add nuw nsw i32 %inner_iv, 1
  %inner_cond = icmp ult i32 %inner_iv_next, 31
  br i1 %inner_cond, label %inner_loop, label %range_check_block

exit:                                            ; preds = %loop_latch, %loop_header
  ret void

range_check_block:                                          ; preds = %inner_loop
  %range_check = icmp slt i32 %iv, 331
  br i1 %range_check, label %loop_latch, label %deopt

loop_latch:                                         ; preds = %range_check_block
  %iv_next = add i32 %iv, 1
  %loop_cond = icmp ult i32 %iv_next, 400
  br i1 %loop_cond, label %loop_header, label %exit

deopt:                                          ; preds = %range_check_block
  ret void
}
