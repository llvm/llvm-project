; RUN: opt -verify-loop-info -irce-print-changed-loops -passes=irce -S < %s 2>&1 | FileCheck %s

; CHECK-LABEL: irce: in function test_01: constrained Loop at depth 2 containing:

define void @test_01(i64 %len) {

; CHECK-LABEL: @test_01(

entry:
  br label %loop

check:
  %entry_check = icmp eq i32 %idx.next, 0
  br i1 %entry_check, label %exit, label %loop

loop:
  %idx = phi i32 [ 1, %entry ], [ %idx.next, %check ]
  %idx_ext = sext i32 %idx to i64
  br label %inner_loop

inner_loop:
  %iv = phi i64 [ 0, %loop ], [ %iv.next, %inner_backedge ]
  %iv.next = add nuw nsw i64 %iv, 1
  %inner_check = icmp slt i64 %iv.next, %idx_ext
  br i1 %inner_check, label %inner, label %outer_check

inner:
  %iv_next_check = icmp slt i64 %iv.next, 100
  br i1 %iv_next_check, label %inner_backedge, label %exit

inner_backedge:
  %cond = icmp eq i64 %iv.next, 100
  br i1 %cond, label %exit, label %inner_loop

outer_check:
  %idx.next = add i32 %idx, 1
  %loopdone = icmp slt i32 %idx.next, 2
  br i1 %loopdone, label %check, label %exit

exit:
  ret void
}
