; RUN: opt -S -jump-threading < %s | FileCheck %s

define i8 @test(i32 %a, i32 %length) {
; CHECK-LABEL: @test
entry:
; CHECK: br label %backedge
  br label %loop

loop:
; CHECK-LABEL: backedge:
; CHECK: phi i32
; CHECK: br i1 %cont, label %backedge, label %exit
  %iv = phi i32 [0, %entry], [%iv.next, %backedge]
  ;; We can use an inductive argument to prove %iv is always positive
  %cnd = icmp sge i32 %iv, 0
  br i1 %cnd, label %backedge, label %exit

backedge:
  %iv.next = add nsw i32 %iv, 1
  %cont = icmp slt i32 %iv.next, 400
  br i1 %cont, label %loop, label %exit
exit:
  ret i8 0
}

