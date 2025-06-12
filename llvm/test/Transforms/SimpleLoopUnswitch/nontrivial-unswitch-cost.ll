; Specifically exercise the cost modeling for non-trivial loop unswitching.
;
; RUN: opt -passes='loop(simple-loop-unswitch<nontrivial>),verify<loops>' -unswitch-threshold=5 -S < %s | FileCheck %s
; RUN: opt -passes='loop-mssa(simple-loop-unswitch<nontrivial>),verify<loops>' -unswitch-threshold=5 -S -verify-memoryssa < %s | FileCheck %s

declare void @a()
declare void @b()
declare void @x()

; First establish enough code size in the duplicated 'loop_begin' block to
; suppress unswitching.
define void @test_no_unswitch(ptr %ptr, i1 %cond) {
; CHECK-LABEL: @test_no_unswitch(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin
;
; We shouldn't have unswitched into any other block either.
; CHECK-NOT:     br i1 %cond

loop_begin:
  call void @x()
  call void @x()
  call void @x()
  call void @x()
  br i1 %cond, label %loop_a, label %loop_b
; CHECK:       loop_begin:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    br i1 %cond, label %loop_a, label %loop_b

loop_a:
  call void @a()
  br label %loop_latch

loop_b:
  call void @b()
  br label %loop_latch

loop_latch:
  %v = load i1, ptr %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret void
}

; Now check that the smaller formulation of 'loop_begin' does in fact unswitch
; with our low threshold.
define void @test_unswitch(ptr %ptr, i1 %cond) {
; CHECK-LABEL: @test_unswitch(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[FROZEN:%.+]] = freeze i1 %cond
; CHECK-NEXT:    br i1 [[FROZEN]], label %entry.split.us, label %entry.split

loop_begin:
  call void @x()
  br i1 %cond, label %loop_a, label %loop_b

loop_a:
  call void @a()
  br label %loop_latch
; The 'loop_a' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    br label %loop_a.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    call void @a()
; CHECK-NEXT:    br label %loop_latch.us
;
; CHECK:       loop_latch.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    br label %loop_exit

loop_b:
  call void @b()
  br label %loop_latch
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    br label %loop_b
;
; CHECK:       loop_b:
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    br label %loop_latch
;
; CHECK:       loop_latch:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit

loop_latch:
  %v = load i1, ptr %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret void
; CHECK:       loop_exit:
; CHECK-NEXT:    ret void
}

; Check that even with large amounts of code on either side of the unswitched
; branch, if that code would be kept in only one of the unswitched clones it
; doesn't contribute to the cost.
define void @test_unswitch_non_dup_code(ptr %ptr, i1 %cond) {
; CHECK-LABEL: @test_unswitch_non_dup_code(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[FROZEN:%.+]] = freeze i1 %cond
; CHECK-NEXT:    br i1 [[FROZEN]], label %entry.split.us, label %entry.split

loop_begin:
  call void @x()
  br i1 %cond, label %loop_a, label %loop_b

loop_a:
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  br label %loop_latch
; The 'loop_a' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    br label %loop_a.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    call void @a()
; CHECK-NEXT:    call void @a()
; CHECK-NEXT:    call void @a()
; CHECK-NEXT:    call void @a()
; CHECK-NEXT:    br label %loop_latch.us
;
; CHECK:       loop_latch.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    br label %loop_exit

loop_b:
  call void @b()
  call void @b()
  call void @b()
  call void @b()
  br label %loop_latch
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    br label %loop_b
;
; CHECK:       loop_b:
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    br label %loop_latch
;
; CHECK:       loop_latch:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit

loop_latch:
  %v = load i1, ptr %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret void
; CHECK:       loop_exit:
; CHECK-NEXT:    ret void
}

; Much like with non-duplicated code directly in the successor, we also won't
; duplicate even interesting CFGs.
define void @test_unswitch_non_dup_code_in_cfg(ptr %ptr, i1 %cond) {
; CHECK-LABEL: @test_unswitch_non_dup_code_in_cfg(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[FROZEN:%.+]] = freeze i1 %cond
; CHECK-NEXT:    br i1 [[FROZEN]], label %entry.split.us, label %entry.split

loop_begin:
  call void @x()
  br i1 %cond, label %loop_a, label %loop_b

loop_a:
  %v1 = load i1, ptr %ptr
  br i1 %v1, label %loop_a_a, label %loop_a_b

loop_a_a:
  call void @a()
  br label %loop_latch

loop_a_b:
  call void @a()
  br label %loop_latch
; The 'loop_a' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    br label %loop_a.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_a_a.us, label %loop_a_b.us
;
; CHECK:       loop_a_b.us:
; CHECK-NEXT:    call void @a()
; CHECK-NEXT:    br label %loop_latch.us
;
; CHECK:       loop_a_a.us:
; CHECK-NEXT:    call void @a()
; CHECK-NEXT:    br label %loop_latch.us
;
; CHECK:       loop_latch.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    br label %loop_exit

loop_b:
  %v2 = load i1, ptr %ptr
  br i1 %v2, label %loop_b_a, label %loop_b_b

loop_b_a:
  call void @b()
  br label %loop_latch

loop_b_b:
  call void @b()
  br label %loop_latch
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    br label %loop_b
;
; CHECK:       loop_b:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_b_a, label %loop_b_b
;
; CHECK:       loop_b_a:
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    br label %loop_latch
;
; CHECK:       loop_b_b:
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    br label %loop_latch
;
; CHECK:       loop_latch:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit

loop_latch:
  %v3 = load i1, ptr %ptr
  br i1 %v3, label %loop_begin, label %loop_exit

loop_exit:
  ret void
; CHECK:       loop_exit:
; CHECK-NEXT:    ret void
}

; Check that even if there is *some* non-duplicated code on one side of an
; unswitch, we don't count any other code in the loop that will in fact have to
; be duplicated.
define void @test_no_unswitch_non_dup_code(ptr %ptr, i1 %cond) {
; CHECK-LABEL: @test_no_unswitch_non_dup_code(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin
;
; We shouldn't have unswitched into any other block either.
; CHECK-NOT:     br i1 %cond

loop_begin:
  call void @x()
  br i1 %cond, label %loop_a, label %loop_b
; CHECK:       loop_begin:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    br i1 %cond, label %loop_a, label %loop_b

loop_a:
  %v1 = load i1, ptr %ptr
  br i1 %v1, label %loop_a_a, label %loop_a_b

loop_a_a:
  call void @a()
  br label %loop_latch

loop_a_b:
  call void @a()
  br label %loop_latch

loop_b:
  %v2 = load i1, ptr %ptr
  br i1 %v2, label %loop_b_a, label %loop_b_b

loop_b_a:
  call void @b()
  br label %loop_latch

loop_b_b:
  call void @b()
  br label %loop_latch

loop_latch:
  call void @x()
  call void @x()
  %v = load i1, ptr %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret void
}

; Check that we still unswitch when the exit block contains lots of code, even
; though we do clone the exit block as part of unswitching. This should work
; because we should split the exit block before anything inside it.
define void @test_unswitch_large_exit(ptr %ptr, i1 %cond) {
; CHECK-LABEL: @test_unswitch_large_exit(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[FROZEN:%.+]] = freeze i1 %cond
; CHECK-NEXT:    br i1 [[FROZEN]], label %entry.split.us, label %entry.split

loop_begin:
  call void @x()
  br i1 %cond, label %loop_a, label %loop_b

loop_a:
  call void @a()
  br label %loop_latch
; The 'loop_a' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    br label %loop_a.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    call void @a()
; CHECK-NEXT:    br label %loop_latch.us
;
; CHECK:       loop_latch.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    br label %loop_exit

loop_b:
  call void @b()
  br label %loop_latch
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    br label %loop_b
;
; CHECK:       loop_b:
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    br label %loop_latch
;
; CHECK:       loop_latch:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit

loop_latch:
  %v = load i1, ptr %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  call void @x()
  call void @x()
  call void @x()
  call void @x()
  ret void
; CHECK:       loop_exit:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    ret void
}

; Check that we handle a dedicated exit edge unswitch which is still
; non-trivial and has lots of code in the exit.
define void @test_unswitch_dedicated_exiting(ptr %ptr, i1 %cond) {
; CHECK-LABEL: @test_unswitch_dedicated_exiting(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[FROZEN:%.+]] = freeze i1 %cond
; CHECK-NEXT:    br i1 [[FROZEN]], label %entry.split.us, label %entry.split

loop_begin:
  call void @x()
  br i1 %cond, label %loop_a, label %loop_b_exit

loop_a:
  call void @a()
  br label %loop_latch
; The 'loop_a' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    br label %loop_a.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    call void @a()
; CHECK-NEXT:    br label %loop_latch.us
;
; CHECK:       loop_latch.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, ptr %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    br label %loop_exit

loop_b_exit:
  call void @b()
  call void @b()
  call void @b()
  call void @b()
  ret void
; The 'loop_b_exit' unswitched exit path.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    call void @x()
; CHECK-NEXT:    br label %loop_b_exit
;
; CHECK:       loop_b_exit:
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    ret void

loop_latch:
  %v = load i1, ptr %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret void
; CHECK:       loop_exit:
; CHECK-NEXT:    ret void
}
