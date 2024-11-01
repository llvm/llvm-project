; RUN: opt -verify-loop-info -irce-print-changed-loops -passes=irce -S < %s 2>&1 | FileCheck %s
; RUN: opt -verify-loop-info -irce-print-changed-loops -passes='require<branch-prob>,irce' -S < %s 2>&1 | FileCheck %s

; CHECK-LABEL: irce: in function test_01: constrained Loop at depth 1 containing:
; CHECK-LABEL: irce: in function test_02: constrained Loop at depth 1 containing:
; CHECK-LABEL: irce: in function test_03: constrained Loop at depth 1 containing:
; CHECK-LABEL: irce: in function test_04: constrained Loop at depth 1 containing:
; CHECK-LABEL: irce: in function test_05: constrained Loop at depth 1 containing:

; This test used to demonstrate a miscompile: the outer loop's IV iterates in
; range of [2, 400) and the range check is done against value 331. Due to a bug
; in range intersection IRCE manages to eliminate the range check without
; inserting a postloop, which is incorrect. We treat the range of this test as
; an unsigned range and are able to intersect ranges correctly and insert a
; postloop.

define void @test_01() {

; CHECK-LABEL: test_01
; CHECK-NOT:     preloop
; CHECK:         range_check_block:                                ; preds = %inner_loop
; CHECK-NEXT:      %range_check = icmp slt i32 %iv, 331
; CHECK-NEXT:      br i1 true, label %loop_latch
; CHECK:         loop_latch:
; CHECK-NEXT:      %iv_next = add i32 %iv, 1
; CHECK-NEXT:      %loop_cond = icmp ult i32 %iv_next, 400
; CHECK-NEXT:      [[COND:%[^ ]+]] = icmp ult i32 %iv_next, 331
; CHECK-NEXT:      br i1 [[COND]], label %loop_header, label %main.exit.selector
; CHECK:         main.exit.selector:                               ; preds = %loop_latch
; CHECK-NEXT:      %iv_next.lcssa = phi i32 [ %iv_next, %loop_latch ]
; CHECK-NEXT:      %iv.lcssa = phi i32 [ %iv, %loop_latch ]
; CHECK-NEXT:      [[MES_COND:%[^ ]+]] = icmp ult i32 %iv_next.lcssa, 400
; CHECK-NEXT:      br i1 [[MES_COND]], label %main.pseudo.exit, label %exit
; CHECK:         loop_latch.postloop:                              ; preds = %range_check_block.postloop
; CHECK-NEXT:      %iv_next.postloop = add i32 %iv.postloop, 1
; CHECK-NEXT:      %loop_cond.postloop = icmp ult i32 %iv_next.postloop, 400
; CHECK-NEXT:      br i1 %loop_cond.postloop, label %loop_header.postloop, label %exit.loopexit

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

; Similar to test_01, but here the range check is done against 450. No postloop
; is required.

define void @test_02() {

; CHECK-LABEL: test_02
; CHECK-NOT:     preloop
; CHECK-NOT:     postloop
; CHECK:         range_check_block:                                ; preds = %inner_loop
; CHECK-NEXT:      %range_check = icmp slt i32 %iv, 450
; CHECK-NEXT:      br i1 true, label %loop_latch
; CHECK:         loop_latch:                                       ; preds = %range_check_block
; CHECK-NEXT:      %iv_next = add i32 %iv, 1
; CHECK-NEXT:      %loop_cond = icmp ult i32 %iv_next, 400
; CHECK-NEXT:      br i1 %loop_cond, label %loop_header, label %exit

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
  %range_check = icmp slt i32 %iv, 450
  br i1 %range_check, label %loop_latch, label %deopt

loop_latch:                                         ; preds = %range_check_block
  %iv_next = add i32 %iv, 1
  %loop_cond = icmp ult i32 %iv_next, 400
  br i1 %loop_cond, label %loop_header, label %exit

deopt:                                          ; preds = %range_check_block
  ret void
}

; Range check is made against 0, so the safe iteration range is empty. IRCE
; should not apply to the inner loop. The condition %tmp2 can be eliminated.

define void @test_03() {

; CHECK-LABEL: test_03
; CHECK-NOT:   preloop
; CHECK-NOT:   postloop
; CHECK:         %tmp2 = icmp sgt i32 %iv.prev, -1
; CHECK-NEXT:    br i1 true, label %loop_header.split.us, label %exit
; CHECK:       range_check_block:
; CHECK-NEXT:    %range_check = icmp slt i32 %iv, 0
; CHECK-NEXT:    br i1 %range_check, label %loop_latch, label %deopt

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
  %range_check = icmp slt i32 %iv, 0
  br i1 %range_check, label %loop_latch, label %deopt

loop_latch:                                         ; preds = %range_check_block
  %iv_next = add i32 %iv, 1
  %loop_cond = icmp ult i32 %iv_next, 400
  br i1 %loop_cond, label %loop_header, label %exit

deopt:                                          ; preds = %range_check_block
  ret void
}

; We can also properly eliminate range check against %n which is not always
; known positive.

define void @test_04(ptr %p) {

; CHECK-LABEL: test_04
; CHECK:       entry
; CHECK-NOT:   preloop
; CHECK:         %tmp2 = icmp sgt i32 %iv.prev, -1
; CHECK-NEXT:    br i1 true, label %loop_header.split.us, label %exit
; CHECK:       range_check_block:
; CHECK-NEXT:    %range_check = icmp slt i32 %iv, %n
; CHECK-NEXT:    br i1 true, label %loop_latch, label %deopt
; CHECK:       postloop:

entry:
  %n = load i32, ptr %p
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
  %range_check = icmp slt i32 %iv, %n
  br i1 %range_check, label %loop_latch, label %deopt

loop_latch:                                         ; preds = %range_check_block
  %iv_next = add i32 %iv, 1
  %loop_cond = icmp ult i32 %iv_next, 400
  br i1 %loop_cond, label %loop_header, label %exit

deopt:                                          ; preds = %range_check_block
  ret void
}

; Same as test_04, but range guarantees that %n is positive. So we can safely
; intersect ranges (with insertion of postloop).

define void @test_05(ptr %p) {

; CHECK-LABEL: test_05
; CHECK-NOT:     preloop
; CHECK:         entry:
; CHECK-NEXT:      %n = load i32, ptr %p, align 4, !range !
; CHECK-NEXT:      %exit.mainloop.at = call i32 @llvm.umax.i32(i32 %n, i32 2)
; CHECK-NEXT:      [[CMP_2:%[^ ]+]] = icmp ult i32 2, %exit.mainloop.at
; CHECK-NEXT:      br i1 [[CMP_2]], label %loop_header.preheader, label %main.pseudo.exit
; CHECK:         range_check_block:                                ; preds = %inner_loop
; CHECK-NEXT:      %range_check = icmp slt i32 %iv, %n
; CHECK-NEXT:      br i1 true, label %loop_latch, label %deopt.loopexit2
; CHECK:         loop_latch:                                       ; preds = %range_check_block
; CHECK-NEXT:      %iv_next = add i32 %iv, 1
; CHECK-NEXT:      %loop_cond = icmp ult i32 %iv_next, 400
; CHECK-NEXT:      [[COND:%[^ ]+]] = icmp ult i32 %iv_next, %exit.mainloop.at
; CHECK-NEXT:      br i1 [[COND]], label %loop_header, label %main.exit.selector
; CHECK:         main.exit.selector:                               ; preds = %loop_latch
; CHECK-NEXT:      %iv_next.lcssa = phi i32 [ %iv_next, %loop_latch ]
; CHECK-NEXT:      %iv.lcssa = phi i32 [ %iv, %loop_latch ]
; CHECK-NEXT:      [[MES_COND:%[^ ]+]] = icmp ult i32 %iv_next.lcssa, 400
; CHECK-NEXT:      br i1 [[MES_COND]], label %main.pseudo.exit, label %exit
; CHECK:         loop_latch.postloop:                              ; preds = %range_check_block.postloop
; CHECK-NEXT:      %iv_next.postloop = add i32 %iv.postloop, 1
; CHECK-NEXT:      %loop_cond.postloop = icmp ult i32 %iv_next.postloop, 400
; CHECK-NEXT:      br i1 %loop_cond.postloop, label %loop_header.postloop, label %exit.loopexit

entry:
  %n = load i32, ptr %p, !range !0
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
  %range_check = icmp slt i32 %iv, %n
  br i1 %range_check, label %loop_latch, label %deopt

loop_latch:                                         ; preds = %range_check_block
  %iv_next = add i32 %iv, 1
  %loop_cond = icmp ult i32 %iv_next, 400
  br i1 %loop_cond, label %loop_header, label %exit

deopt:                                          ; preds = %range_check_block
  ret void
}

!0 = !{i32 0, i32 50}
