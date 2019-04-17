; RUN: opt -passes='print<scalar-evolution>,loop(unswitch,loop-instsimplify),print<scalar-evolution>' -enable-nontrivial-unswitch -S < %s 2>%t.scev | FileCheck %s
; RUN: opt -enable-mssa-loop-dependency=true -verify-memoryssa -passes='print<scalar-evolution>,loop(unswitch,loop-instsimplify),print<scalar-evolution>' -enable-nontrivial-unswitch -S < %s 2>%t.scev | FileCheck %s
; RUN: FileCheck %s --check-prefix=SCEV < %t.scev

target triple = "x86_64-unknown-linux-gnu"

declare void @f()

; Check that trivially unswitching an inner loop resets both the inner and outer
; loop trip count.
define void @test1(i32 %n, i32 %m, i1 %cond) {
; Check that SCEV has no trip count before unswitching.
; SCEV-LABEL: Determining loop execution counts for: @test1
; SCEV: Loop %inner_loop_begin: <multiple exits> Unpredictable backedge-taken count.
; SCEV: Loop %outer_loop_begin: Unpredictable backedge-taken count.
;
; Now check that after unswitching and simplifying instructions we get clean
; backedge-taken counts.
; SCEV-LABEL: Determining loop execution counts for: @test1
; SCEV: Loop %inner_loop_begin: backedge-taken count is (-1 + (1 smax %m))<nsw>
; SCEV: Loop %outer_loop_begin: backedge-taken count is (-1 + (1 smax %n))<nsw>
;
; And verify the code matches what we expect.
; CHECK-LABEL: define void @test1(
entry:
  br label %outer_loop_begin
; Ensure the outer loop didn't get unswitched.
; CHECK:       entry:
; CHECK-NEXT:    br label %outer_loop_begin

outer_loop_begin:
  %i = phi i32 [ %i.next, %outer_loop_latch ], [ 0, %entry ]
  ; Block unswitching of the outer loop with a noduplicate call.
  call void @f() noduplicate
  br label %inner_loop_begin
; Ensure the inner loop got unswitched into the outer loop.
; CHECK:       outer_loop_begin:
; CHECK-NEXT:    %{{.*}} = phi i32
; CHECK-NEXT:    call void @f()
; CHECK-NEXT:    br i1 %cond,

inner_loop_begin:
  %j = phi i32 [ %j.next, %inner_loop_latch ], [ 0, %outer_loop_begin ]
  br i1 %cond, label %inner_loop_latch, label %inner_loop_early_exit

inner_loop_latch:
  %j.next = add nsw i32 %j, 1
  %j.cmp = icmp slt i32 %j.next, %m
  br i1 %j.cmp, label %inner_loop_begin, label %inner_loop_late_exit

inner_loop_early_exit:
  %j.lcssa = phi i32 [ %i, %inner_loop_begin ]
  br label %outer_loop_latch

inner_loop_late_exit:
  br label %outer_loop_latch

outer_loop_latch:
  %i.phi = phi i32 [ %j.lcssa, %inner_loop_early_exit ], [ %i, %inner_loop_late_exit ]
  %i.next = add nsw i32 %i.phi, 1
  %i.cmp = icmp slt i32 %i.next, %n
  br i1 %i.cmp, label %outer_loop_begin, label %exit

exit:
  ret void
}

; Check that trivially unswitching an inner loop resets both the inner and outer
; loop trip count.
define void @test2(i32 %n, i32 %m, i32 %cond) {
; Check that SCEV has no trip count before unswitching.
; SCEV-LABEL: Determining loop execution counts for: @test2
; SCEV: Loop %inner_loop_begin: <multiple exits> Unpredictable backedge-taken count.
; SCEV: Loop %outer_loop_begin: Unpredictable backedge-taken count.
;
; Now check that after unswitching and simplifying instructions we get clean
; backedge-taken counts.
; SCEV-LABEL: Determining loop execution counts for: @test2
; SCEV: Loop %inner_loop_begin: backedge-taken count is (-1 + (1 smax %m))<nsw>
; SCEV: Loop %outer_loop_begin: backedge-taken count is (-1 + (1 smax %n))<nsw>
;
; CHECK-LABEL: define void @test2(
entry:
  br label %outer_loop_begin
; Ensure the outer loop didn't get unswitched.
; CHECK:       entry:
; CHECK-NEXT:    br label %outer_loop_begin

outer_loop_begin:
  %i = phi i32 [ %i.next, %outer_loop_latch ], [ 0, %entry ]
  ; Block unswitching of the outer loop with a noduplicate call.
  call void @f() noduplicate
  br label %inner_loop_begin
; Ensure the inner loop got unswitched into the outer loop.
; CHECK:       outer_loop_begin:
; CHECK-NEXT:    %{{.*}} = phi i32
; CHECK-NEXT:    call void @f()
; CHECK-NEXT:    switch i32 %cond,

inner_loop_begin:
  %j = phi i32 [ %j.next, %inner_loop_latch ], [ 0, %outer_loop_begin ]
  switch i32 %cond, label %inner_loop_early_exit [
    i32 1, label %inner_loop_latch
    i32 2, label %inner_loop_latch
  ]

inner_loop_latch:
  %j.next = add nsw i32 %j, 1
  %j.cmp = icmp slt i32 %j.next, %m
  br i1 %j.cmp, label %inner_loop_begin, label %inner_loop_late_exit

inner_loop_early_exit:
  %j.lcssa = phi i32 [ %i, %inner_loop_begin ]
  br label %outer_loop_latch

inner_loop_late_exit:
  br label %outer_loop_latch

outer_loop_latch:
  %i.phi = phi i32 [ %j.lcssa, %inner_loop_early_exit ], [ %i, %inner_loop_late_exit ]
  %i.next = add nsw i32 %i.phi, 1
  %i.cmp = icmp slt i32 %i.next, %n
  br i1 %i.cmp, label %outer_loop_begin, label %exit

exit:
  ret void
}

; Check that non-trivial unswitching of a branch in an inner loop into the outer
; loop invalidates both inner and outer.
define void @test3(i32 %n, i32 %m, i1 %cond) {
; Check that SCEV has no trip count before unswitching.
; SCEV-LABEL: Determining loop execution counts for: @test3
; SCEV: Loop %inner_loop_begin: <multiple exits> Unpredictable backedge-taken count.
; SCEV: Loop %outer_loop_begin: Unpredictable backedge-taken count.
;
; Now check that after unswitching and simplifying instructions we get clean
; backedge-taken counts.
; SCEV-LABEL: Determining loop execution counts for: @test3
; SCEV: Loop %inner_loop_begin{{.*}}: backedge-taken count is (-1 + (1 smax %m))<nsw>
; SCEV: Loop %outer_loop_begin: backedge-taken count is (-1 + (1 smax %n))<nsw>
;
; And verify the code matches what we expect.
; CHECK-LABEL: define void @test3(
entry:
  br label %outer_loop_begin
; Ensure the outer loop didn't get unswitched.
; CHECK:       entry:
; CHECK-NEXT:    br label %outer_loop_begin

outer_loop_begin:
  %i = phi i32 [ %i.next, %outer_loop_latch ], [ 0, %entry ]
  ; Block unswitching of the outer loop with a noduplicate call.
  call void @f() noduplicate
  br label %inner_loop_begin
; Ensure the inner loop got unswitched into the outer loop.
; CHECK:       outer_loop_begin:
; CHECK-NEXT:    %{{.*}} = phi i32
; CHECK-NEXT:    call void @f()
; CHECK-NEXT:    br i1 %cond,

inner_loop_begin:
  %j = phi i32 [ %j.next, %inner_loop_latch ], [ 0, %outer_loop_begin ]
  %j.tmp = add nsw i32 %j, 1
  br i1 %cond, label %inner_loop_latch, label %inner_loop_early_exit

inner_loop_latch:
  %j.next = add nsw i32 %j, 1
  %j.cmp = icmp slt i32 %j.next, %m
  br i1 %j.cmp, label %inner_loop_begin, label %inner_loop_late_exit

inner_loop_early_exit:
  %j.lcssa = phi i32 [ %j.tmp, %inner_loop_begin ]
  br label %outer_loop_latch

inner_loop_late_exit:
  br label %outer_loop_latch

outer_loop_latch:
  %inc.phi = phi i32 [ %j.lcssa, %inner_loop_early_exit ], [ 1, %inner_loop_late_exit ]
  %i.next = add nsw i32 %i, %inc.phi
  %i.cmp = icmp slt i32 %i.next, %n
  br i1 %i.cmp, label %outer_loop_begin, label %exit

exit:
  ret void
}
