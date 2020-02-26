; RUN: opt -passes='loop(unswitch<nontrivial>),verify<loops>' -simple-loop-unswitch-guards -S < %s | FileCheck %s
; RUN: opt -simple-loop-unswitch -enable-nontrivial-unswitch -simple-loop-unswitch-guards -S < %s | FileCheck %s
; RUN: opt -passes='loop-mssa(unswitch<nontrivial>),verify<loops>' -simple-loop-unswitch-guards  -verify-memoryssa -S < %s | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

define void @test_simple_case(i1 %cond, i32 %N) {
; CHECK-LABEL: @test_simple_case(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[ENTRY_SPLIT_US:%.*]], label [[ENTRY_SPLIT:%.*]]
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label [[LOOP_US:%.*]]
; CHECK:       loop.us:
; CHECK-NEXT:    [[IV_US:%.*]] = phi i32 [ 0, [[ENTRY_SPLIT_US]] ], [ [[IV_NEXT_US:%.*]], [[GUARDED_US:%.*]] ]
; CHECK-NEXT:    br label [[GUARDED_US]]
; CHECK:       guarded.us:
; CHECK-NEXT:    [[IV_NEXT_US]] = add i32 [[IV_US]], 1
; CHECK-NEXT:    [[LOOP_COND_US:%.*]] = icmp slt i32 [[IV_NEXT_US]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[LOOP_COND_US]], label [[LOOP_US]], label [[EXIT_SPLIT_US:%.*]]
; CHECK:       deopt:
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"() ]
; CHECK-NEXT:    unreachable
;

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
  %iv.next = add i32 %iv, 1
  %loop.cond = icmp slt i32 %iv.next, %N
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void
}

define void @test_two_guards(i1 %cond1, i1 %cond2, i32 %N) {
; CHECK-LABEL: @test_two_guards(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND1:%.*]], label [[ENTRY_SPLIT_US:%.*]], label [[ENTRY_SPLIT:%.*]]
; CHECK:       entry.split.us:
; CHECK-NEXT:    br i1 [[COND2:%.*]], label [[ENTRY_SPLIT_US_SPLIT_US:%.*]], label [[ENTRY_SPLIT_US_SPLIT:%.*]]
; CHECK:       entry.split.us.split.us:
; CHECK-NEXT:    br label [[LOOP_US_US:%.*]]
; CHECK:       loop.us.us:
; CHECK-NEXT:    [[IV_US_US:%.*]] = phi i32 [ 0, [[ENTRY_SPLIT_US_SPLIT_US]] ], [ [[IV_NEXT_US_US:%.*]], [[GUARDED_US2:%.*]] ]
; CHECK-NEXT:    br label [[GUARDED_US_US:%.*]]
; CHECK:       guarded.us.us:
; CHECK-NEXT:    br label [[GUARDED_US2]]
; CHECK:       guarded.us2:
; CHECK-NEXT:    [[IV_NEXT_US_US]] = add i32 [[IV_US_US]], 1
; CHECK-NEXT:    [[LOOP_COND_US_US:%.*]] = icmp slt i32 [[IV_NEXT_US_US]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[LOOP_COND_US_US]], label [[LOOP_US_US]], label [[EXIT_SPLIT_US_SPLIT_US:%.*]]
; CHECK:       deopt1:
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"() ]
; CHECK-NEXT:    unreachable
; CHECK:       deopt:
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"() ]
; CHECK-NEXT:    unreachable
; CHECK:       exit:
; CHECK-NEXT:    ret void
;

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cond2) [ "deopt"() ]
  %iv.next = add i32 %iv, 1
  %loop.cond = icmp slt i32 %iv.next, %N
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void
}

define void @test_conditional_guards(i1 %cond, i32 %N) {
; CHECK-LABEL: @test_conditional_guards(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[COND_FR:%.*]] = freeze i1 [[COND:%.*]]
; CHECK-NEXT:    br i1 [[COND_FR]], label [[ENTRY_SPLIT_US:%.*]], label [[ENTRY_SPLIT:%.*]]
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label [[LOOP_US:%.*]]
; CHECK:       loop.us:
; CHECK-NEXT:    [[IV_US:%.*]] = phi i32 [ 0, [[ENTRY_SPLIT_US]] ], [ [[IV_NEXT_US:%.*]], [[BACKEDGE_US:%.*]] ]
; CHECK-NEXT:    [[CONDITION_US:%.*]] = icmp eq i32 [[IV_US]], 123
; CHECK-NEXT:    br i1 [[CONDITION_US]], label [[GUARD_US:%.*]], label [[BACKEDGE_US]]
; CHECK:       guard.us:
; CHECK-NEXT:    br label [[GUARDED_US:%.*]]
; CHECK:       backedge.us:
; CHECK-NEXT:    [[IV_NEXT_US]] = add i32 [[IV_US]], 1
; CHECK-NEXT:    [[LOOP_COND_US:%.*]] = icmp slt i32 [[IV_NEXT_US]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[LOOP_COND_US]], label [[LOOP_US]], label [[EXIT_SPLIT_US:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ 0, [[ENTRY_SPLIT]] ], [ [[IV_NEXT:%.*]], [[BACKEDGE:%.*]] ]
; CHECK-NEXT:    [[CONDITION:%.*]] = icmp eq i32 [[IV]], 123
; CHECK-NEXT:    br i1 [[CONDITION]], label [[GUARD:%.*]], label [[BACKEDGE]]
; CHECK:       guard:
; CHECK-NEXT:    br label [[DEOPT:%.*]]
; CHECK:       deopt:
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"() ]
; CHECK-NEXT:    unreachable
; CHECK:       backedge:
; CHECK-NEXT:    [[IV_NEXT]] = add i32 [[IV]], 1
; CHECK-NEXT:    [[LOOP_COND:%.*]] = icmp slt i32 [[IV_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[LOOP_COND]], label %loop, label [[EXIT_SPLIT:%.*]]
;

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %condition = icmp eq i32 %iv, 123
  br i1 %condition, label %guard, label %backedge

guard:
  call void (i1, ...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
  br label %backedge

backedge:
  %iv.next = add i32 %iv, 1
  %loop.cond = icmp slt i32 %iv.next, %N
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void
}

define void @test_nested_loop(i1 %cond, i32 %N) {
; CHECK-LABEL: @test_nested_loop(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[ENTRY_SPLIT:%.*]], label [[OUTER_LOOP_SPLIT:%.*]]
; CHECK:       entry.split:
; CHECK-NEXT:    br label [[OUTER_LOOP:%.*]]
; CHECK:       outer_loop:
; CHECK-NEXT:    br label [[OUTER_LOOP_SPLIT_US:%.*]]
; CHECK:       outer_loop.split.us:
; CHECK-NEXT:    br label [[LOOP_US:%.*]]
; CHECK:       loop.us:
; CHECK-NEXT:    [[IV_US:%.*]] = phi i32 [ 0, [[OUTER_LOOP_SPLIT_US]] ], [ [[IV_NEXT_US:%.*]], [[GUARDED_US:%.*]] ]
; CHECK-NEXT:    br label [[GUARDED_US]]
; CHECK:       guarded.us:
; CHECK-NEXT:    [[IV_NEXT_US]] = add i32 [[IV_US]], 1
; CHECK-NEXT:    [[LOOP_COND_US:%.*]] = icmp slt i32 [[IV_NEXT_US]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[LOOP_COND_US]], label [[LOOP_US]], label [[OUTER_BACKEDGE_SPLIT_US:%.*]]
; CHECK:       outer_backedge.split.us:
; CHECK-NEXT:    br label [[OUTER_BACKEDGE:%.*]]
; CHECK:       deopt:
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"() ]
; CHECK-NEXT:    unreachable
; CHECK:       outer_backedge:
; CHECK-NEXT:    br i1 false, label [[OUTER_LOOP]], label [[EXIT:%.*]]
;

entry:
  br label %outer_loop

outer_loop:
  br label %loop

loop:
  %iv = phi i32 [ 0, %outer_loop ], [ %iv.next, %loop ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
  %iv.next = add i32 %iv, 1
  %loop.cond = icmp slt i32 %iv.next, %N
  br i1 %loop.cond, label %loop, label %outer_backedge

outer_backedge:
  br i1 undef, label %outer_loop, label %exit

exit:
  ret void
}

define void @test_sibling_loops(i1 %cond1, i1 %cond2, i32 %N) {
; CHECK-LABEL: @test_sibling_loops(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND1:%.*]], label [[ENTRY_SPLIT_US:%.*]], label [[ENTRY_SPLIT:%.*]]
; CHECK:         [[IV1_US:%.*]] = phi i32 [ 0, [[ENTRY_SPLIT_US]] ], [ [[IV1_NEXT_US:%.*]], [[GUARDED_US:%.*]] ]
; CHECK-NEXT:    br label [[GUARDED_US]]
; CHECK:         call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"() ]
; CHECK-NEXT:    unreachable
; CHECK:         [[IV2_US:%.*]] = phi i32 [ 0, [[BETWEEN:%.*]] ], [ [[IV1_NEXT_US2:%.*]], [[GUARDED_US2:%.*]] ]
; CHECK-NEXT:    br label [[GUARDED_US2]]
; CHECK:         call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"() ]
; CHECK-NEXT:    unreachable
;

entry:
  br label %loop1

loop1:
  %iv1 = phi i32 [ 0, %entry ], [ %iv1.next, %loop1 ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]
  %iv1.next = add i32 %iv1, 1
  %loop1.cond = icmp slt i32 %iv1.next, %N
  br i1 %loop1.cond, label %loop1, label %between

between:
  br label %loop2

loop2:
  %iv2 = phi i32 [ 0, %between ], [ %iv2.next, %loop2 ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cond2) [ "deopt"() ]
  %iv2.next = add i32 %iv2, 1
  %loop2.cond = icmp slt i32 %iv2.next, %N
  br i1 %loop2.cond, label %loop2, label %exit

exit:
  ret void
}

; Check that we don't do anything because of cleanuppad.
; CHECK-LABEL: @test_cleanuppad(
; CHECK:       call void (i1, ...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
; CHECK-NOT:   call void (i1, ...) @llvm.experimental.guard(
define void @test_cleanuppad(i1 %cond, i32 %N) personality i32 (...)* @__CxxFrameHandler3 {

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
  %iv.next = add i32 %iv, 1
  invoke void @may_throw(i32 %iv) to label %loop unwind label %exit

exit:
  %cp = cleanuppad within none []
  cleanupret from %cp unwind to caller

}

declare void @may_throw(i32 %i)
declare i32 @__CxxFrameHandler3(...)
