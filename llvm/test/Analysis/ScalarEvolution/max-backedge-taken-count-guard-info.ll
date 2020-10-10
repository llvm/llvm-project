; RUN: opt -analyze -scalar-evolution %s -enable-new-pm=0 | FileCheck %s
; RUN: opt -passes='print<scalar-evolution>' -disable-output %s 2>&1 | FileCheck %s

; Test case for PR40961. The loop guard limit the max backedge-taken count.

define void @test_guard_less_than_16(i32* nocapture %a, i64 %i) {
; CHECK-LABEL: Determining loop execution counts for: @test_guard_less_than_16
; CHECK-NEXT:  Loop %loop: backedge-taken count is (15 + (-1 * %i))
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 15
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

define void @test_guard_less_than_16_operands_swapped(i32* nocapture %a, i64 %i) {
; CHECK-LABEL: Determining loop execution counts for: @test_guard_less_than_16_operands_swapped
; CHECK-NEXT:  Loop %loop: backedge-taken count is (15 + (-1 * %i))
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 15
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is (15 + (-1 * %i))
;
entry:
  %cmp3 = icmp ugt i64 16, %i
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

define void @test_guard_less_than_16_branches_flipped(i32* nocapture %a, i64 %i) {
; CHECK-LABEL: Determining loop execution counts for: @test_guard_less_than_16_branches_flipped
; CHECK-NEXT:  Loop %loop: backedge-taken count is (15 + (-1 * %i))
; CHECK-NEXT:  Loop %loop: max backedge-taken count is -1
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is (15 + (-1 * %i))
;
entry:
  %cmp3 = icmp ult i64 %i, 16
  br i1 %cmp3, label %exit, label %loop

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

define void @test_guard_uge_16_branches_flipped(i32* nocapture %a, i64 %i) {
; CHECK-LABEL: Determining loop execution counts for: @test_guard_uge_16_branches_flipped
; CHECK-NEXT:  Loop %loop: backedge-taken count is (15 + (-1 * %i))
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 15
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is (15 + (-1 * %i))
;
entry:
  %cmp3 = icmp uge i64 %i, 16
  br i1 %cmp3, label %exit, label %loop

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

define void @test_guard_eq_12(i32* nocapture %a, i64 %N) {
; CHECK-LABEL: Determining loop execution counts for: @test_guard_eq_12
; CHECK-NEXT:  Loop %loop: backedge-taken count is %N
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 12
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is %N
;
entry:
  %c.1 = icmp eq i64 %N, 12
  br i1 %c.1, label %loop, label %exit

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %idx = getelementptr inbounds i32, i32* %a, i64 %iv
  store i32 1, i32* %idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv, %N
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @test_guard_ule_12(i32* nocapture %a, i64 %N) {
; CHECK-LABEL: Determining loop execution counts for: @test_guard_ule_12
; CHECK-NEXT:  Loop %loop: backedge-taken count is %N
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 12
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is %N
;
entry:
  %c.1 = icmp ule i64 %N, 12
  br i1 %c.1, label %loop, label %exit

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %idx = getelementptr inbounds i32, i32* %a, i64 %iv
  store i32 1, i32* %idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv, %N
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @test_guard_ule_12_step2(i32* nocapture %a, i64 %N) {
; CHECK-LABEL: Determining loop execution counts for: @test_guard_ule_12_step2
; CHECK-NEXT:  Loop %loop: backedge-taken count is (%N /u 2)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 9223372036854775807
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is (%N /u 2)
;
entry:
  %c.1 = icmp ule i64 %N, 12
  br i1 %c.1, label %loop, label %exit

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %idx = getelementptr inbounds i32, i32* %a, i64 %iv
  store i32 1, i32* %idx, align 4
  %iv.next = add nuw nsw i64 %iv, 2
  %exitcond = icmp eq i64 %iv, %N
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @test_multiple_const_guards_order1(i32* nocapture %a, i64 %i) {
; CHECK-LABEL: @test_multiple_const_guards_order1
; CHECK:       Loop %loop: backedge-taken count is %i
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 9
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is %i
;
entry:
  %c.1 = icmp ult i64 %i, 16
  br i1 %c.1, label %guardbb, label %exit

guardbb:
  %c.2 = icmp ult i64 %i, 10
  br i1 %c.2, label %loop, label %exit

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %guardbb ]
  %idx = getelementptr inbounds i32, i32* %a, i64 %iv
  store i32 1, i32* %idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv, %i
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @test_multiple_const_guards_order2(i32* nocapture %a, i64 %i) {
; CHECK-LABEL: @test_multiple_const_guards_order2
; CHECK:       Loop %loop: backedge-taken count is %i
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 9
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is %i
;
entry:
  %c.1 = icmp ult i64 %i, 10
  br i1 %c.1, label %guardbb, label %exit

guardbb:
  %c.2 = icmp ult i64 %i, 16
  br i1 %c.2, label %loop, label %exit

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %guardbb ]
  %idx = getelementptr inbounds i32, i32* %a, i64 %iv
  store i32 1, i32* %idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv, %i
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; TODO: Currently we miss getting the tightest max backedge-taken count (11).
define void @test_multiple_var_guards_order1(i32* nocapture %a, i64 %i, i64 %N) {
; CHECK-LABEL: @test_multiple_var_guards_order1
; CHECK:       Loop %loop: backedge-taken count is %i
; CHECK-NEXT:  Loop %loop: max backedge-taken count is -1
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is %i
;
entry:
  %c.1 = icmp ult i64 %N, 12
  br i1 %c.1, label %guardbb, label %exit

guardbb:
  %c.2 = icmp ult i64 %i, %N
  br i1 %c.2, label %loop, label %exit

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %guardbb ]
  %idx = getelementptr inbounds i32, i32* %a, i64 %iv
  store i32 1, i32* %idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv, %i
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; TODO: Currently we miss getting the tightest max backedge-taken count (11).
define void @test_multiple_var_guards_order2(i32* nocapture %a, i64 %i, i64 %N) {
; CHECK-LABEL: Determining loop execution counts for: @test_multiple_var_guards_order2
; CHECK-NEXT:  Loop %loop: backedge-taken count is %i
; CHECK-NEXT:  Loop %loop: max backedge-taken count is -1
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is %i
;
entry:
  %c.1 = icmp ult i64 %i, %N
  br i1 %c.1, label %guardbb, label %exit

guardbb:
  %c.2 = icmp ult i64 %N, 12
  br i1 %c.2, label %loop, label %exit

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %guardbb ]
  %idx = getelementptr inbounds i32, i32* %a, i64 %iv
  store i32 1, i32* %idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv, %i
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; The guards here reference each other in a cycle.
define void @test_multiple_var_guards_cycle(i32* nocapture %a, i64 %i, i64 %N) {
; CHECK-LABEL: Determining loop execution counts for: @test_multiple_var_guards_cycle
; CHECK-NEXT:  Loop %loop: backedge-taken count is %N
; CHECK-NEXT:  Loop %loop: max backedge-taken count is -1
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is %N
;
entry:
  %c.1 = icmp ult i64 %N, %i
  br i1 %c.1, label %guardbb, label %exit

guardbb:
  %c.2 = icmp ult i64 %i, %N
  br i1 %c.2, label %loop, label %exit

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %guardbb ]
  %idx = getelementptr inbounds i32, i32* %a, i64 %iv
  store i32 1, i32* %idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv, %N
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @test_guard_ult_ne(i32* nocapture readonly %data, i64 %count) {
; CHECK-LABEL: @test_guard_ult_ne
; CHECK:       Loop %loop: backedge-taken count is (-1 + %count)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 3
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is (-1 + %count)
;
entry:
  %cmp.ult = icmp ult i64 %count, 5
  br i1 %cmp.ult, label %guardbb, label %exit

guardbb:
  %cmp.ne = icmp ne i64 %count, 0
  br i1 %cmp.ne, label %loop, label %exit

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %guardbb ]
  %idx = getelementptr inbounds i32, i32* %data, i64 %iv
  store i32 1, i32* %idx, align 4
  %iv.next = add nuw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %count
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

; Test case for PR47247. Both the guard condition and the assume limit the
; max backedge-taken count.

define void @test_guard_and_assume(i32* nocapture readonly %data, i64 %count) {
; CHECK-LABEL: @test_guard_and_assume
; CHECK:       Loop %loop: backedge-taken count is (-1 + %count)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 3
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

define void @guard_pessimizes_analysis(i1 %c, i32 %N) {
; CHECK-LABEL: @guard_pessimizes_analysis
; CHECK:      Loop %loop: backedge-taken count is (9 + (-1 * %init)<nsw>)<nsw>
; CHECK-NEXT: Loop %loop: max backedge-taken count is 7
; CHECK-NEXT: Loop %loop: Predicated backedge-taken count is (9 + (-1 * %init)<nsw>)<nsw>
;
entry:
  br i1 %c, label %bb1, label %guard

bb1:
  br label %guard

guard:
  %init = phi i32 [ 2, %entry ], [ 3, %bb1 ]
  %c.1 = icmp ult i32 %init, %N
  br i1 %c.1, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ %init, %loop.ph ]
  %iv.next = add i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, 10
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @crash(i8* %ptr) {
; CHECK-LABEL: @crash
; CHECK:       Loop %while.body125: backedge-taken count is {(-2 + (-1 * %ptr)),+,-1}<nw><%while.cond111>
; CHECK-NEXT:  Loop %while.body125: max backedge-taken count is -1
; CHECK-NEXT:  Loop %while.body125: Predicated backedge-taken count is {(-2 + (-1 * %ptr)),+,-1}<nw><%while.cond111>
;
entry:
  br label %while.body

while.body:
  br label %while.cond111

while.cond111:
  %text.addr.5 = phi i8* [ %incdec.ptr112, %while.cond111 ], [ null, %while.body ]
  %incdec.ptr112 = getelementptr inbounds i8, i8* %text.addr.5, i64 -1
  br i1 false, label %while.end117, label %while.cond111

while.end117:
  %cmp118 = icmp ult i8* %ptr, %incdec.ptr112
  br i1 %cmp118, label %while.body125, label %while.cond134.preheader


while.cond134.preheader:
  br label %while.body

while.body125:
  %lastout.2271 = phi i8* [ %incdec.ptr126, %while.body125 ], [ %ptr, %while.end117 ]
  %incdec.ptr126 = getelementptr inbounds i8, i8* %lastout.2271, i64 1
  %exitcond.not = icmp eq i8* %incdec.ptr126, %incdec.ptr112
  br i1 %exitcond.not, label %while.end129, label %while.body125

while.end129:                                     ; preds = %while.body125
  ret void
}
