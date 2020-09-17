; RUN: opt -analyze -scalar-evolution %s | FileCheck %s

; Test case for PR40961. The loop guard limit the max backedge-taken count.

define void @test_guard_less_than_16(i32* nocapture %a, i64 %i) {
; CHECK-LABEL: Determining loop execution counts for: @test_guard_less_than_16
; CHECK-NEXT:  Loop %loop: backedge-taken count is (15 + (-1 * %i))
; CHECK-NEXT:  Loop %loop: max backedge-taken count is -1
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is (15 + (-1 * %i))
;
entry:
  %cmp3 = icmp ult i64 %i, 16
  br i1 %cmp3, label %loop, label %exit

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ %i, %entry ]
  %idx = getelementptr inbounds i32, i32* %a, i64 %iv
  store i32 1, i32* %idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 16
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; Test case for PR47247. Both the guard condition and the assume limit the
; max backedge-taken count.

define void @test_guard_and_assume(i32* nocapture readonly %data, i64 %count) {
; CHECK-LABEL: Determining loop execution counts for: @test_guard_and_assume
; CHECK-NEXT:  Loop %loop: backedge-taken count is (-1 + %count)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is -2
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is (-1 + %count)
;
entry:
  %cmp = icmp ult i64 %count, 5
  tail call void @llvm.assume(i1 %cmp)
  %cmp18.not = icmp eq i64 %count, 0
  br i1 %cmp18.not, label %exit, label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %idx = getelementptr inbounds i32, i32* %data, i64 %iv
  store i32 1, i32* %idx, align 4
  %iv.next = add nuw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %count
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1 noundef)
