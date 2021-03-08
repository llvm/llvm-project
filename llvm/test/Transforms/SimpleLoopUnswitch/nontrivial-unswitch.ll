; RUN: opt -passes='loop(unswitch<nontrivial>),verify<loops>' -S < %s | FileCheck %s
; RUN: opt -passes='loop-mssa(unswitch<nontrivial>),verify<loops>' -S < %s | FileCheck %s
; RUN: opt -simple-loop-unswitch -enable-nontrivial-unswitch -S < %s | FileCheck %s
; RUN: opt -simple-loop-unswitch -enable-nontrivial-unswitch -enable-mssa-loop-dependency=true -verify-memoryssa -S < %s | FileCheck %s

declare i32 @a()
declare i32 @b()
declare i32 @c()
declare i32 @d()

declare void @sink1(i32)
declare void @sink2(i32)

declare i1 @cond()
declare i32 @cond.i32()

; Negative test: we cannot unswitch convergent calls.
define void @test_no_unswitch_convergent(i1* %ptr, i1 %cond) {
; CHECK-LABEL: @test_no_unswitch_convergent(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin
;
; We shouldn't have unswitched into any other block either.
; CHECK-NOT:     br i1 %cond

loop_begin:
  br i1 %cond, label %loop_a, label %loop_b
; CHECK:       loop_begin:
; CHECK-NEXT:    br i1 %cond, label %loop_a, label %loop_b

loop_a:
  call i32 @a() convergent
  br label %loop_latch

loop_b:
  call i32 @b()
  br label %loop_latch

loop_latch:
  %v = load i1, i1* %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret void
}

; Negative test: we cannot unswitch noduplicate calls.
define void @test_no_unswitch_noduplicate(i1* %ptr, i1 %cond) {
; CHECK-LABEL: @test_no_unswitch_noduplicate(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin
;
; We shouldn't have unswitched into any other block either.
; CHECK-NOT:     br i1 %cond

loop_begin:
  br i1 %cond, label %loop_a, label %loop_b
; CHECK:       loop_begin:
; CHECK-NEXT:    br i1 %cond, label %loop_a, label %loop_b

loop_a:
  call i32 @a() noduplicate
  br label %loop_latch

loop_b:
  call i32 @b()
  br label %loop_latch

loop_latch:
  %v = load i1, i1* %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret void
}

declare i32 @__CxxFrameHandler3(...)

; Negative test: we cannot unswitch when tokens are used across blocks as we
; might introduce PHIs.
define void @test_no_unswitch_cross_block_token(i1* %ptr, i1 %cond) nounwind personality i32 (...)* @__CxxFrameHandler3 {
; CHECK-LABEL: @test_no_unswitch_cross_block_token(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin
;
; We shouldn't have unswitched into any other block either.
; CHECK-NOT:     br i1 %cond

loop_begin:
  br i1 %cond, label %loop_a, label %loop_b
; CHECK:       loop_begin:
; CHECK-NEXT:    br i1 %cond, label %loop_a, label %loop_b

loop_a:
  call i32 @a()
  br label %loop_cont

loop_b:
  call i32 @b()
  br label %loop_cont

loop_cont:
  invoke i32 @a()
          to label %loop_latch unwind label %loop_catch

loop_latch:
  br label %loop_begin

loop_catch:
  %catch = catchswitch within none [label %loop_catch_latch, label %loop_exit] unwind to caller

loop_catch_latch:
  %catchpad_latch = catchpad within %catch []
  catchret from %catchpad_latch to label %loop_begin

loop_exit:
  %catchpad_exit = catchpad within %catch []
  catchret from %catchpad_exit to label %exit

exit:
  ret void
}


; Non-trivial loop unswitching where there are two distinct trivial conditions
; to unswitch within the loop.
define i32 @test1(i1* %ptr, i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @test1(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %entry.split.us, label %entry.split

loop_begin:
  br i1 %cond1, label %loop_a, label %loop_b

loop_a:
  call i32 @a()
  br label %latch
; The 'loop_a' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    br label %loop_a.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    call i32 @a()
; CHECK-NEXT:    br label %latch.us
;
; CHECK:       latch.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    br label %loop_exit

loop_b:
  br i1 %cond2, label %loop_b_a, label %loop_b_b
; The second unswitched condition.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br i1 %cond2, label %entry.split.split.us, label %entry.split.split

loop_b_a:
  call i32 @b()
  br label %latch
; The 'loop_b_a' unswitched loop.
;
; CHECK:       entry.split.split.us:
; CHECK-NEXT:    br label %loop_begin.us1
;
; CHECK:       loop_begin.us1:
; CHECK-NEXT:    br label %loop_b.us
;
; CHECK:       loop_b.us:
; CHECK-NEXT:    br label %loop_b_a.us
;
; CHECK:       loop_b_a.us:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %latch.us2
;
; CHECK:       latch.us2:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us1, label %loop_exit.split.split.us
;
; CHECK:       loop_exit.split.split.us:
; CHECK-NEXT:    br label %loop_exit.split

loop_b_b:
  call i32 @c()
  br label %latch
; The 'loop_b_b' unswitched loop.
;
; CHECK:       entry.split.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %loop_b
;
; CHECK:       loop_b:
; CHECK-NEXT:    br label %loop_b_b
;
; CHECK:       loop_b_b:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %latch
;
; CHECK:       latch:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit.split.split
;
; CHECK:       loop_exit.split.split:
; CHECK-NEXT:    br label %loop_exit.split

latch:
  %v = load i1, i1* %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret i32 0
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    ret
}

define i32 @test2(i1* %ptr, i1 %cond1, i32* %a.ptr, i32* %b.ptr, i32* %c.ptr) {
; CHECK-LABEL: @test2(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %entry.split.us, label %entry.split

loop_begin:
  %v = load i1, i1* %ptr
  br i1 %cond1, label %loop_a, label %loop_b

loop_a:
  %a = load i32, i32* %a.ptr
  %ac = load i32, i32* %c.ptr
  br i1 %v, label %loop_begin, label %loop_exit
; The 'loop_a' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br label %loop_a.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[AC:.*]] = load i32, i32* %c.ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.backedge.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A]], %loop_a.us ]
; CHECK-NEXT:    %[[AC_LCSSA:.*]] = phi i32 [ %[[AC]], %loop_a.us ]
; CHECK-NEXT:    br label %loop_exit

loop_b:
  %b = load i32, i32* %b.ptr
  %bc = load i32, i32* %c.ptr
  br i1 %v, label %loop_begin, label %loop_exit
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br label %loop_b
;
; CHECK:       loop_b:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    %[[BC:.*]] = load i32, i32* %c.ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.backedge, label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %loop_b ]
; CHECK-NEXT:    %[[BC_LCSSA:.*]] = phi i32 [ %[[BC]], %loop_b ]
; CHECK-NEXT:    br label %loop_exit

loop_exit:
  %ab.phi = phi i32 [ %a, %loop_a ], [ %b, %loop_b ]
  %c.phi = phi i32 [ %ac, %loop_a ], [ %bc, %loop_b ]
  %result = add i32 %ab.phi, %c.phi
  ret i32 %result
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[AB_PHI:.*]] = phi i32 [ %[[B_LCSSA]], %loop_exit.split ], [ %[[A_LCSSA]], %loop_exit.split.us ]
; CHECK-NEXT:    %[[C_PHI:.*]] = phi i32 [ %[[BC_LCSSA]], %loop_exit.split ], [ %[[AC_LCSSA]], %loop_exit.split.us ]
; CHECK-NEXT:    %[[RESULT:.*]] = add i32 %[[AB_PHI]], %[[C_PHI]]
; CHECK-NEXT:    ret i32 %[[RESULT]]
}

; Test a non-trivial unswitch of an exiting edge to an exit block with other
; in-loop predecessors.
define i32 @test3a(i1* %ptr, i1 %cond1, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test3a(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %entry.split.us, label %entry.split

loop_begin:
  %v = load i1, i1* %ptr
  %a = load i32, i32* %a.ptr
  br i1 %cond1, label %loop_exit, label %loop_b
; The 'loop_exit' clone.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A]], %loop_begin.us ]
; CHECK-NEXT:    br label %loop_exit

loop_b:
  %b = load i32, i32* %b.ptr
  br i1 %v, label %loop_begin, label %loop_exit
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_b
;
; CHECK:       loop_b:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %loop_b ]
; CHECK-NEXT:    br label %loop_exit

loop_exit:
  %ab.phi = phi i32 [ %a, %loop_begin ], [ %b, %loop_b ]
  ret i32 %ab.phi
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[AB_PHI:.*]] = phi i32 [ %[[B_LCSSA]], %loop_exit.split ], [ %[[A_LCSSA]], %loop_exit.split.us ]
; CHECK-NEXT:    ret i32 %[[AB_PHI]]
}

; Test a non-trivial unswitch of an exiting edge to an exit block with other
; in-loop predecessors. This is the same as @test3a but with the reversed order
; of successors so that the exiting edge is *not* the cloned edge.
define i32 @test3b(i1* %ptr, i1 %cond1, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test3b(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %entry.split.us, label %entry.split

loop_begin:
  %v = load i1, i1* %ptr
  %a = load i32, i32* %a.ptr
  br i1 %cond1, label %loop_b, label %loop_exit
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_b.us
;
; CHECK:       loop_b.us:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %loop_b.us ]
; CHECK-NEXT:    br label %loop_exit

loop_b:
  %b = load i32, i32* %b.ptr
  br i1 %v, label %loop_begin, label %loop_exit
; The original loop, now non-looping due to unswitching..
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit

loop_exit:
  %ab.phi = phi i32 [ %b, %loop_b ], [ %a, %loop_begin ]
  ret i32 %ab.phi
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[AB_PHI:.*]] = phi i32 [ %[[A]], %loop_exit.split ], [ %[[B_LCSSA]], %loop_exit.split.us ]
; CHECK-NEXT:    ret i32 %[[AB_PHI]]
}

; Test a non-trivial unswitch of an exiting edge to an exit block with no other
; in-loop predecessors.
define void @test4a(i1* %ptr, i1 %cond1, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test4a(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %entry.split.us, label %entry.split

loop_begin:
  %v = load i1, i1* %ptr
  %a = load i32, i32* %a.ptr
  br i1 %cond1, label %loop_exit1, label %loop_b
; The 'loop_exit' clone.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_exit1.split.us
;
; CHECK:       loop_exit1.split.us:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A]], %loop_begin.us ]
; CHECK-NEXT:    br label %loop_exit1

loop_b:
  %b = load i32, i32* %b.ptr
  br i1 %v, label %loop_begin, label %loop_exit2
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_b
;
; CHECK:       loop_b:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit2

loop_exit1:
  %a.phi = phi i32 [ %a, %loop_begin ]
  call void @sink1(i32 %a.phi)
  ret void
; CHECK:       loop_exit1:
; CHECK-NEXT:    call void @sink1(i32 %[[A_LCSSA]])
; CHECK-NEXT:    ret void

loop_exit2:
  %b.phi = phi i32 [ %b, %loop_b ]
  call void @sink2(i32 %b.phi)
  ret void
; CHECK:       loop_exit2:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %loop_b ]
; CHECK-NEXT:    call void @sink2(i32 %[[B_LCSSA]])
; CHECK-NEXT:    ret void
}

; Test a non-trivial unswitch of an exiting edge to an exit block with no other
; in-loop predecessors. This is the same as @test4a but with the edges reversed
; so that the exiting edge is *not* the cloned edge.
define void @test4b(i1* %ptr, i1 %cond1, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test4b(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %entry.split.us, label %entry.split

loop_begin:
  %v = load i1, i1* %ptr
  %a = load i32, i32* %a.ptr
  br i1 %cond1, label %loop_b, label %loop_exit1
; The 'loop_b' clone.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_b.us
;
; CHECK:       loop_b.us:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us, label %loop_exit2.split.us
;
; CHECK:       loop_exit2.split.us:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %loop_b.us ]
; CHECK-NEXT:    br label %loop_exit2

loop_b:
  %b = load i32, i32* %b.ptr
  br i1 %v, label %loop_begin, label %loop_exit2
; The 'loop_exit' unswitched path.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_exit1

loop_exit1:
  %a.phi = phi i32 [ %a, %loop_begin ]
  call void @sink1(i32 %a.phi)
  ret void
; CHECK:       loop_exit1:
; CHECK-NEXT:    %[[A_PHI:.*]] = phi i32 [ %[[A]], %loop_begin ]
; CHECK-NEXT:    call void @sink1(i32 %[[A_PHI]])
; CHECK-NEXT:    ret void

loop_exit2:
  %b.phi = phi i32 [ %b, %loop_b ]
  call void @sink2(i32 %b.phi)
  ret void
; CHECK:       loop_exit2:
; CHECK-NEXT:    call void @sink2(i32 %[[B_LCSSA]])
; CHECK-NEXT:    ret void
}

; Test a non-trivial unswitch of an exiting edge to an exit block with no other
; in-loop predecessors. This is the same as @test4a but with a common merge
; block after the independent loop exits. This requires a different structural
; update to the dominator tree.
define void @test4c(i1* %ptr, i1 %cond1, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test4c(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %entry.split.us, label %entry.split

loop_begin:
  %v = load i1, i1* %ptr
  %a = load i32, i32* %a.ptr
  br i1 %cond1, label %loop_exit1, label %loop_b
; The 'loop_exit' clone.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_exit1.split.us
;
; CHECK:       loop_exit1.split.us:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A]], %loop_begin.us ]
; CHECK-NEXT:    br label %loop_exit1

loop_b:
  %b = load i32, i32* %b.ptr
  br i1 %v, label %loop_begin, label %loop_exit2
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_b
;
; CHECK:       loop_b:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit2

loop_exit1:
  %a.phi = phi i32 [ %a, %loop_begin ]
  call void @sink1(i32 %a.phi)
  br label %exit
; CHECK:       loop_exit1:
; CHECK-NEXT:    call void @sink1(i32 %[[A_LCSSA]])
; CHECK-NEXT:    br label %exit

loop_exit2:
  %b.phi = phi i32 [ %b, %loop_b ]
  call void @sink2(i32 %b.phi)
  br label %exit
; CHECK:       loop_exit2:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %loop_b ]
; CHECK-NEXT:    call void @sink2(i32 %[[B_LCSSA]])
; CHECK-NEXT:    br label %exit

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; Test that we can unswitch a condition out of multiple layers of a loop nest.
define i32 @test5(i1* %ptr, i1 %cond1, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test5(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %loop_begin.split.us, label %entry.split
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %loop_begin.split

loop_begin:
  br label %inner_loop_begin

inner_loop_begin:
  %v = load i1, i1* %ptr
  %a = load i32, i32* %a.ptr
  br i1 %cond1, label %loop_exit, label %inner_loop_b
; The 'loop_exit' clone.
;
; CHECK:       loop_begin.split.us:
; CHECK-NEXT:    br label %inner_loop_begin.us
;
; CHECK:       inner_loop_begin.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_exit.loopexit.split.us
;
; CHECK:       loop_exit.loopexit.split.us:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A]], %inner_loop_begin.us ]
; CHECK-NEXT:    br label %loop_exit

inner_loop_b:
  %b = load i32, i32* %b.ptr
  br i1 %v, label %inner_loop_begin, label %loop_latch
; The 'inner_loop_b' unswitched loop.
;
; CHECK:       loop_begin.split:
; CHECK-NEXT:    br label %inner_loop_begin
;
; CHECK:       inner_loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_b
;
; CHECK:       inner_loop_b:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_begin, label %loop_latch

loop_latch:
  %b.phi = phi i32 [ %b, %inner_loop_b ]
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %loop_begin, label %loop_exit
; CHECK:       loop_latch:
; CHECK-NEXT:    %[[B_INNER_LCSSA:.*]] = phi i32 [ %[[B]], %inner_loop_b ]
; CHECK-NEXT:    %[[V2:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V2]], label %loop_begin, label %loop_exit.loopexit1

loop_exit:
  %ab.phi = phi i32 [ %a, %inner_loop_begin ], [ %b.phi, %loop_latch ]
  ret i32 %ab.phi
; CHECK:       loop_exit.loopexit:
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit.loopexit1:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B_INNER_LCSSA]], %loop_latch ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[AB_PHI:.*]] = phi i32 [ %[[A_LCSSA]], %loop_exit.loopexit ], [ %[[B_LCSSA]], %loop_exit.loopexit1 ]
; CHECK-NEXT:    ret i32 %[[AB_PHI]]
}

; Test that we can unswitch a condition where we end up only cloning some of
; the nested loops and needing to delete some of the nested loops.
define i32 @test6(i1* %ptr, i1 %cond1, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test6(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %entry.split.us, label %entry.split

loop_begin:
  %v = load i1, i1* %ptr
  br i1 %cond1, label %loop_a, label %loop_b

loop_a:
  br label %loop_a_inner

loop_a_inner:
  %va = load i1, i1* %ptr
  %a = load i32, i32* %a.ptr
  br i1 %va, label %loop_a_inner, label %loop_a_inner_exit

loop_a_inner_exit:
  %a.lcssa = phi i32 [ %a, %loop_a_inner ]
  br label %latch
; The 'loop_a' cloned loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br label %loop_a.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    br label %loop_a_inner.us
;
; CHECK:       loop_a_inner.us
; CHECK-NEXT:    %[[VA:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br i1 %[[VA]], label %loop_a_inner.us, label %loop_a_inner_exit.us
;
; CHECK:       loop_a_inner_exit.us:
; CHECK-NEXT:    %[[A_INNER_LCSSA:.*]] = phi i32 [ %[[A]], %loop_a_inner.us ]
; CHECK-NEXT:    br label %latch.us
;
; CHECK:       latch.us:
; CHECK-NEXT:    %[[A_PHI:.*]] = phi i32 [ %[[A_INNER_LCSSA]], %loop_a_inner_exit.us ]
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A_PHI]], %latch.us ]
; CHECK-NEXT:    br label %loop_exit

loop_b:
  br label %loop_b_inner

loop_b_inner:
  %vb = load i1, i1* %ptr
  %b = load i32, i32* %b.ptr
  br i1 %vb, label %loop_b_inner, label %loop_b_inner_exit

loop_b_inner_exit:
  %b.lcssa = phi i32 [ %b, %loop_b_inner ]
  br label %latch

latch:
  %ab.phi = phi i32 [ %a.lcssa, %loop_a_inner_exit ], [ %b.lcssa, %loop_b_inner_exit ]
  br i1 %v, label %loop_begin, label %loop_exit
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br label %loop_b
;
; CHECK:       loop_b:
; CHECK-NEXT:    br label %loop_b_inner
;
; CHECK:       loop_b_inner
; CHECK-NEXT:    %[[VB:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br i1 %[[VB]], label %loop_b_inner, label %loop_b_inner_exit
;
; CHECK:       loop_b_inner_exit:
; CHECK-NEXT:    %[[B_INNER_LCSSA:.*]] = phi i32 [ %[[B]], %loop_b_inner ]
; CHECK-NEXT:    br label %latch
;
; CHECK:       latch:
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B_INNER_LCSSA]], %latch ]
; CHECK-NEXT:    br label %loop_exit

loop_exit:
  %ab.lcssa = phi i32 [ %ab.phi, %latch ]
  ret i32 %ab.lcssa
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[AB_PHI:.*]] = phi i32 [ %[[B_LCSSA]], %loop_exit.split ], [ %[[A_LCSSA]], %loop_exit.split.us ]
; CHECK-NEXT:    ret i32 %[[AB_PHI]]
}

; Test that when unswitching a deeply nested loop condition in a way that
; produces a non-loop clone that can reach multiple exit blocks which are part
; of different outer loops we correctly divide the cloned loop blocks between
; the outer loops based on reachability.
define i32 @test7a(i1* %ptr, i1* %cond.ptr, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test7a(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %a = load i32, i32* %a.ptr
  br label %inner_loop_begin
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_begin

inner_loop_begin:
  %a.phi = phi i32 [ %a, %loop_begin ], [ %a2, %inner_inner_loop_exit ]
  %cond = load i1, i1* %cond.ptr
  %b = load i32, i32* %b.ptr
  br label %inner_inner_loop_begin
; CHECK:       inner_loop_begin:
; CHECK-NEXT:    %[[A_INNER_PHI:.*]] = phi i32 [ %[[A]], %loop_begin ], [ %[[A2:.*]], %inner_inner_loop_exit ]
; CHECK-NEXT:    %[[COND:.*]] = load i1, i1* %cond.ptr
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br i1 %[[COND]], label %inner_loop_begin.split.us, label %inner_loop_begin.split

inner_inner_loop_begin:
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %inner_inner_loop_a, label %inner_inner_loop_b

inner_inner_loop_a:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %loop_exit, label %inner_inner_loop_c

inner_inner_loop_b:
  %v3 = load i1, i1* %ptr
  br i1 %v3, label %inner_inner_loop_exit, label %inner_inner_loop_c

inner_inner_loop_c:
  %v4 = load i1, i1* %ptr
  br i1 %v4, label %inner_loop_exit, label %inner_inner_loop_d

inner_inner_loop_d:
  br i1 %cond, label %inner_loop_exit, label %inner_inner_loop_begin
; The cloned copy that always exits with the adjustments required to fix up
; loop exits.
;
; CHECK:       inner_loop_begin.split.us:
; CHECK-NEXT:    br label %inner_inner_loop_begin.us
;
; CHECK:       inner_inner_loop_begin.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_a.us, label %inner_inner_loop_b.us
;
; CHECK:       inner_inner_loop_b.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_exit.split.us, label %inner_inner_loop_c.us.loopexit
;
; CHECK:       inner_inner_loop_a.us:
; CHECK-NEXT:    %[[A_NEW_LCSSA:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_inner_loop_begin.us ]
; CHECK-NEXT:    %[[B_NEW_LCSSA:.*]] = phi i32 [ %[[B]], %inner_inner_loop_begin.us ]
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_exit.split.us, label %inner_inner_loop_c.us
;
; CHECK:       inner_inner_loop_c.us.loopexit:
; CHECK-NEXT:    br label %inner_inner_loop_c.us
;
; CHECK:       inner_inner_loop_c.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_exit.loopexit.split.us, label %inner_inner_loop_d.us
;
; CHECK:       inner_inner_loop_d.us:
; CHECK-NEXT:    br label %inner_loop_exit.loopexit.split
;
; CHECK:       inner_inner_loop_exit.split.us:
; CHECK-NEXT:    br label %inner_inner_loop_exit
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    %[[A_LCSSA_US:.*]] = phi i32 [ %[[A_NEW_LCSSA]], %inner_inner_loop_a.us ]
; CHECK-NEXT:    %[[B_LCSSA_US:.*]] = phi i32 [ %[[B_NEW_LCSSA]], %inner_inner_loop_a.us ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       inner_loop_exit.loopexit.split.us:
; CHECK-NEXT:    br label %inner_loop_exit.loopexit
;
; The original copy that continues to loop.
;
; CHECK:       inner_loop_begin.split:
; CHECK-NEXT:    br label %inner_inner_loop_begin
;
; CHECK:       inner_inner_loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_a, label %inner_inner_loop_b
;
; CHECK:       inner_inner_loop_a:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_exit.split, label %inner_inner_loop_c
;
; CHECK:       inner_inner_loop_b:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_exit.split, label %inner_inner_loop_c
;
; CHECK:       inner_inner_loop_c:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_exit.loopexit.split, label %inner_inner_loop_d
;
; CHECK:       inner_inner_loop_d:
; CHECK-NEXT:    br label %inner_inner_loop_begin
;
; CHECK:       inner_inner_loop_exit.split:
; CHECK-NEXT:    br label %inner_inner_loop_exit

inner_inner_loop_exit:
  %a2 = load i32, i32* %a.ptr
  %v5 = load i1, i1* %ptr
  br i1 %v5, label %inner_loop_exit, label %inner_loop_begin
; CHECK:       inner_inner_loop_exit:
; CHECK-NEXT:    %[[A2]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_exit.loopexit1, label %inner_loop_begin

inner_loop_exit:
  br label %loop_begin
; CHECK:       inner_loop_exit.loopexit.split:
; CHECK-NEXT:    br label %inner_loop_exit.loopexit
;
; CHECK:       inner_loop_exit.loopexit:
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit.loopexit1:
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit:
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  %a.lcssa = phi i32 [ %a.phi, %inner_inner_loop_a ]
  %b.lcssa = phi i32 [ %b, %inner_inner_loop_a ]
  %result = add i32 %a.lcssa, %b.lcssa
  ret i32 %result
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_inner_loop_a ]
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %inner_inner_loop_a ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_PHI:.*]] = phi i32 [ %[[A_LCSSA]], %loop_exit.split ], [ %[[A_LCSSA_US]], %loop_exit.split.us ]
; CHECK-NEXT:    %[[B_PHI:.*]] = phi i32 [ %[[B_LCSSA]], %loop_exit.split ], [ %[[B_LCSSA_US]], %loop_exit.split.us ]
; CHECK-NEXT:    %[[RESULT:.*]] = add i32 %[[A_PHI]], %[[B_PHI]]
; CHECK-NEXT:    ret i32 %[[RESULT]]
}

; Same pattern as @test7a but here the original loop becomes a non-loop that
; can reach multiple exit blocks which are part of different outer loops.
define i32 @test7b(i1* %ptr, i1* %cond.ptr, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test7b(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %a = load i32, i32* %a.ptr
  br label %inner_loop_begin
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_begin

inner_loop_begin:
  %a.phi = phi i32 [ %a, %loop_begin ], [ %a2, %inner_inner_loop_exit ]
  %cond = load i1, i1* %cond.ptr
  %b = load i32, i32* %b.ptr
  br label %inner_inner_loop_begin
; CHECK:       inner_loop_begin:
; CHECK-NEXT:    %[[A_INNER_PHI:.*]] = phi i32 [ %[[A]], %loop_begin ], [ %[[A2:.*]], %inner_inner_loop_exit ]
; CHECK-NEXT:    %[[COND:.*]] = load i1, i1* %cond.ptr
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br i1 %[[COND]], label %inner_loop_begin.split.us, label %inner_loop_begin.split

inner_inner_loop_begin:
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %inner_inner_loop_a, label %inner_inner_loop_b

inner_inner_loop_a:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %loop_exit, label %inner_inner_loop_c

inner_inner_loop_b:
  %v3 = load i1, i1* %ptr
  br i1 %v3, label %inner_inner_loop_exit, label %inner_inner_loop_c

inner_inner_loop_c:
  %v4 = load i1, i1* %ptr
  br i1 %v4, label %inner_loop_exit, label %inner_inner_loop_d

inner_inner_loop_d:
  br i1 %cond, label %inner_inner_loop_begin, label %inner_loop_exit
; The cloned copy that continues looping.
;
; CHECK:       inner_loop_begin.split.us:
; CHECK-NEXT:    br label %inner_inner_loop_begin.us
;
; CHECK:       inner_inner_loop_begin.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_a.us, label %inner_inner_loop_b.us
;
; CHECK:       inner_inner_loop_b.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_exit.split.us, label %inner_inner_loop_c.us
;
; CHECK:       inner_inner_loop_a.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_exit.split.us, label %inner_inner_loop_c.us
;
; CHECK:       inner_inner_loop_c.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_exit.loopexit.split.us, label %inner_inner_loop_d.us
;
; CHECK:       inner_inner_loop_d.us:
; CHECK-NEXT:    br label %inner_inner_loop_begin.us
;
; CHECK:       inner_inner_loop_exit.split.us:
; CHECK-NEXT:    br label %inner_inner_loop_exit
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    %[[A_LCSSA_US:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_inner_loop_a.us ]
; CHECK-NEXT:    %[[B_LCSSA_US:.*]] = phi i32 [ %[[B]], %inner_inner_loop_a.us ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       inner_loop_exit.loopexit.split.us:
; CHECK-NEXT:    br label %inner_loop_exit.loopexit
;
; The original copy that now always exits and needs adjustments for exit
; blocks.
;
; CHECK:       inner_loop_begin.split:
; CHECK-NEXT:    br label %inner_inner_loop_begin
;
; CHECK:       inner_inner_loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_a, label %inner_inner_loop_b
;
; CHECK:       inner_inner_loop_a:
; CHECK-NEXT:    %[[A_NEW_LCSSA:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_inner_loop_begin ]
; CHECK-NEXT:    %[[B_NEW_LCSSA:.*]] = phi i32 [ %[[B]], %inner_inner_loop_begin ]
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_exit.split, label %inner_inner_loop_c
;
; CHECK:       inner_inner_loop_b:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_exit.split, label %inner_inner_loop_c.loopexit
;
; CHECK:       inner_inner_loop_c.loopexit:
; CHECK-NEXT:    br label %inner_inner_loop_c
;
; CHECK:       inner_inner_loop_c:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_exit.loopexit.split, label %inner_inner_loop_d
;
; CHECK:       inner_inner_loop_d:
; CHECK-NEXT:    br label %inner_loop_exit.loopexit.split
;
; CHECK:       inner_inner_loop_exit.split:
; CHECK-NEXT:    br label %inner_inner_loop_exit

inner_inner_loop_exit:
  %a2 = load i32, i32* %a.ptr
  %v5 = load i1, i1* %ptr
  br i1 %v5, label %inner_loop_exit, label %inner_loop_begin
; CHECK:       inner_inner_loop_exit:
; CHECK-NEXT:    %[[A2]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_exit.loopexit1, label %inner_loop_begin

inner_loop_exit:
  br label %loop_begin
; CHECK:       inner_loop_exit.loopexit.split:
; CHECK-NEXT:    br label %inner_loop_exit.loopexit
;
; CHECK:       inner_loop_exit.loopexit:
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit.loopexit1:
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit:
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  %a.lcssa = phi i32 [ %a.phi, %inner_inner_loop_a ]
  %b.lcssa = phi i32 [ %b, %inner_inner_loop_a ]
  %result = add i32 %a.lcssa, %b.lcssa
  ret i32 %result
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A_NEW_LCSSA]], %inner_inner_loop_a ]
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B_NEW_LCSSA]], %inner_inner_loop_a ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_PHI:.*]] = phi i32 [ %[[A_LCSSA]], %loop_exit.split ], [ %[[A_LCSSA_US]], %loop_exit.split.us ]
; CHECK-NEXT:    %[[B_PHI:.*]] = phi i32 [ %[[B_LCSSA]], %loop_exit.split ], [ %[[B_LCSSA_US]], %loop_exit.split.us ]
; CHECK-NEXT:    %[[RESULT:.*]] = add i32 %[[A_PHI]], %[[B_PHI]]
; CHECK-NEXT:    ret i32 %[[RESULT]]
}

; Test that when the exit block set of an inner loop changes to start at a less
; high level of the loop nest we correctly hoist the loop up the nest.
define i32 @test8a(i1* %ptr, i1* %cond.ptr, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test8a(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %a = load i32, i32* %a.ptr
  br label %inner_loop_begin
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_begin

inner_loop_begin:
  %a.phi = phi i32 [ %a, %loop_begin ], [ %a2, %inner_inner_loop_exit ]
  %cond = load i1, i1* %cond.ptr
  %b = load i32, i32* %b.ptr
  br label %inner_inner_loop_begin
; CHECK:       inner_loop_begin:
; CHECK-NEXT:    %[[A_INNER_PHI:.*]] = phi i32 [ %[[A]], %loop_begin ], [ %[[A2:.*]], %inner_inner_loop_exit ]
; CHECK-NEXT:    %[[COND:.*]] = load i1, i1* %cond.ptr
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br i1 %[[COND]], label %inner_loop_begin.split.us, label %inner_loop_begin.split

inner_inner_loop_begin:
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %inner_inner_loop_a, label %inner_inner_loop_b

inner_inner_loop_a:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %inner_inner_loop_latch, label %inner_loop_exit

inner_inner_loop_b:
  br i1 %cond, label %inner_inner_loop_latch, label %inner_inner_loop_exit

inner_inner_loop_latch:
  br label %inner_inner_loop_begin
; The cloned region is now an exit from the inner loop.
;
; CHECK:       inner_loop_begin.split.us:
; CHECK-NEXT:    %[[A_INNER_INNER_LCSSA:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_loop_begin ]
; CHECK-NEXT:    br label %inner_inner_loop_begin.us
;
; CHECK:       inner_inner_loop_begin.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_a.us, label %inner_inner_loop_b.us
;
; CHECK:       inner_inner_loop_b.us:
; CHECK-NEXT:    br label %inner_inner_loop_latch.us
;
; CHECK:       inner_inner_loop_a.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_latch.us, label %inner_loop_exit.loopexit.split.us
;
; CHECK:       inner_inner_loop_latch.us:
; CHECK-NEXT:    br label %inner_inner_loop_begin.us
;
; CHECK:       inner_loop_exit.loopexit.split.us:
; CHECK-NEXT:    %[[A_INNER_LCSSA_US:.*]] = phi i32 [ %[[A_INNER_INNER_LCSSA]], %inner_inner_loop_a.us ]
; CHECK-NEXT:    br label %inner_loop_exit.loopexit
;
; The original region exits the loop earlier.
;
; CHECK:       inner_loop_begin.split:
; CHECK-NEXT:    br label %inner_inner_loop_begin
;
; CHECK:       inner_inner_loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_a, label %inner_inner_loop_b
;
; CHECK:       inner_inner_loop_a:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_latch, label %inner_loop_exit.loopexit.split
;
; CHECK:       inner_inner_loop_b:
; CHECK-NEXT:    br label %inner_inner_loop_exit
;
; CHECK:       inner_inner_loop_latch:
; CHECK-NEXT:    br label %inner_inner_loop_begin

inner_inner_loop_exit:
  %a2 = load i32, i32* %a.ptr
  %v4 = load i1, i1* %ptr
  br i1 %v4, label %inner_loop_exit, label %inner_loop_begin
; CHECK:       inner_inner_loop_exit:
; CHECK-NEXT:    %[[A2]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_exit.loopexit1, label %inner_loop_begin

inner_loop_exit:
  %v5 = load i1, i1* %ptr
  br i1 %v5, label %loop_exit, label %loop_begin
; CHECK:       inner_loop_exit.loopexit.split:
; CHECK-NEXT:    %[[A_INNER_LCSSA:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_inner_loop_a ]
; CHECK-NEXT:    br label %inner_loop_exit.loopexit
;
; CHECK:       inner_loop_exit.loopexit:
; CHECK-NEXT:    %[[A_INNER_US_PHI:.*]] = phi i32 [ %[[A_INNER_LCSSA]], %inner_loop_exit.loopexit.split ], [ %[[A_INNER_LCSSA_US]], %inner_loop_exit.loopexit.split.us ]
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit.loopexit1:
; CHECK-NEXT:    %[[A_INNER_LCSSA2:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_inner_loop_exit ]
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit:
; CHECK-NEXT:    %[[A_INNER_PHI:.*]] = phi i32 [ %[[A_INNER_LCSSA2]], %inner_loop_exit.loopexit1 ], [ %[[A_INNER_US_PHI]], %inner_loop_exit.loopexit ]
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_exit, label %loop_begin

loop_exit:
  %a.lcssa = phi i32 [ %a.phi, %inner_loop_exit ]
  ret i32 %a.lcssa
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_loop_exit ]
; CHECK-NEXT:    ret i32 %[[A_LCSSA]]
}

; Same pattern as @test8a but where the original loop looses an exit block and
; needs to be hoisted up the nest.
define i32 @test8b(i1* %ptr, i1* %cond.ptr, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test8b(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %a = load i32, i32* %a.ptr
  br label %inner_loop_begin
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_begin

inner_loop_begin:
  %a.phi = phi i32 [ %a, %loop_begin ], [ %a2, %inner_inner_loop_exit ]
  %cond = load i1, i1* %cond.ptr
  %b = load i32, i32* %b.ptr
  br label %inner_inner_loop_begin
; CHECK:       inner_loop_begin:
; CHECK-NEXT:    %[[A_INNER_PHI:.*]] = phi i32 [ %[[A]], %loop_begin ], [ %[[A2:.*]], %inner_inner_loop_exit ]
; CHECK-NEXT:    %[[COND:.*]] = load i1, i1* %cond.ptr
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br i1 %[[COND]], label %inner_loop_begin.split.us, label %inner_loop_begin.split

inner_inner_loop_begin:
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %inner_inner_loop_a, label %inner_inner_loop_b

inner_inner_loop_a:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %inner_inner_loop_latch, label %inner_loop_exit

inner_inner_loop_b:
  br i1 %cond, label %inner_inner_loop_exit, label %inner_inner_loop_latch

inner_inner_loop_latch:
  br label %inner_inner_loop_begin
; The cloned region is similar to before but with one earlier exit.
;
; CHECK:       inner_loop_begin.split.us:
; CHECK-NEXT:    br label %inner_inner_loop_begin.us
;
; CHECK:       inner_inner_loop_begin.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_a.us, label %inner_inner_loop_b.us
;
; CHECK:       inner_inner_loop_b.us:
; CHECK-NEXT:    br label %inner_inner_loop_exit.split.us
;
; CHECK:       inner_inner_loop_a.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_latch.us, label %inner_loop_exit.loopexit.split.us
;
; CHECK:       inner_inner_loop_latch.us:
; CHECK-NEXT:    br label %inner_inner_loop_begin.us
;
; CHECK:       inner_inner_loop_exit.split.us:
; CHECK-NEXT:    br label %inner_inner_loop_exit
;
; CHECK:       inner_loop_exit.loopexit.split.us:
; CHECK-NEXT:    %[[A_INNER_LCSSA_US:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_inner_loop_a.us ]
; CHECK-NEXT:    br label %inner_loop_exit.loopexit
;
; The original region is now an exit in the preheader.
;
; CHECK:       inner_loop_begin.split:
; CHECK-NEXT:    %[[A_INNER_INNER_LCSSA:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_loop_begin ]
; CHECK-NEXT:    br label %inner_inner_loop_begin
;
; CHECK:       inner_inner_loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_a, label %inner_inner_loop_b
;
; CHECK:       inner_inner_loop_a:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_latch, label %inner_loop_exit.loopexit.split
;
; CHECK:       inner_inner_loop_b:
; CHECK-NEXT:    br label %inner_inner_loop_latch
;
; CHECK:       inner_inner_loop_latch:
; CHECK-NEXT:    br label %inner_inner_loop_begin

inner_inner_loop_exit:
  %a2 = load i32, i32* %a.ptr
  %v4 = load i1, i1* %ptr
  br i1 %v4, label %inner_loop_exit, label %inner_loop_begin
; CHECK:       inner_inner_loop_exit:
; CHECK-NEXT:    %[[A2]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_exit.loopexit1, label %inner_loop_begin

inner_loop_exit:
  %v5 = load i1, i1* %ptr
  br i1 %v5, label %loop_exit, label %loop_begin
; CHECK:       inner_loop_exit.loopexit.split:
; CHECK-NEXT:    %[[A_INNER_LCSSA:.*]] = phi i32 [ %[[A_INNER_INNER_LCSSA]], %inner_inner_loop_a ]
; CHECK-NEXT:    br label %inner_loop_exit.loopexit
;
; CHECK:       inner_loop_exit.loopexit:
; CHECK-NEXT:    %[[A_INNER_US_PHI:.*]] = phi i32 [ %[[A_INNER_LCSSA]], %inner_loop_exit.loopexit.split ], [ %[[A_INNER_LCSSA_US]], %inner_loop_exit.loopexit.split.us ]
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit.loopexit1:
; CHECK-NEXT:    %[[A_INNER_LCSSA2:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_inner_loop_exit ]
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit:
; CHECK-NEXT:    %[[A_INNER_PHI:.*]] = phi i32 [ %[[A_INNER_LCSSA2]], %inner_loop_exit.loopexit1 ], [ %[[A_INNER_US_PHI]], %inner_loop_exit.loopexit ]
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_exit, label %loop_begin

loop_exit:
  %a.lcssa = phi i32 [ %a.phi, %inner_loop_exit ]
  ret i32 %a.lcssa
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_loop_exit ]
; CHECK-NEXT:    ret i32 %[[A_LCSSA]]
}

; Test for when unswitching produces a clone of an inner loop but
; the clone no longer has an exiting edge *at all* and loops infinitely.
; Because it doesn't ever exit to the outer loop it is no longer an inner loop
; but needs to be hoisted up the nest to be a top-level loop.
define i32 @test9a(i1* %ptr, i1* %cond.ptr, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test9a(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %b = load i32, i32* %b.ptr
  %cond = load i1, i1* %cond.ptr
  br label %inner_loop_begin
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    %[[COND:.*]] = load i1, i1* %cond.ptr
; CHECK-NEXT:    br i1 %[[COND]], label %loop_begin.split.us, label %loop_begin.split

inner_loop_begin:
  %a = load i32, i32* %a.ptr
  br i1 %cond, label %inner_loop_latch, label %inner_loop_exit

inner_loop_latch:
  call void @sink1(i32 %b)
  br label %inner_loop_begin
; The cloned inner loop ends up as an infinite loop and thus being a top-level
; loop with the preheader as an exit block of the outer loop.
;
; CHECK:       loop_begin.split.us
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %loop_begin ]
; CHECK-NEXT:    br label %inner_loop_begin.us
;
; CHECK:       inner_loop_begin.us:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_latch.us
;
; CHECK:       inner_loop_latch.us:
; CHECK-NEXT:    call void @sink1(i32 %[[B_LCSSA]])
; CHECK-NEXT:    br label %inner_loop_begin.us
;
; The original loop becomes boring non-loop code.
;
; CHECK:       loop_begin.split
; CHECK-NEXT:    br label %inner_loop_begin
;
; CHECK:       inner_loop_begin:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_exit

inner_loop_exit:
  %a.inner_lcssa = phi i32 [ %a, %inner_loop_begin ]
  %v = load i1, i1* %ptr
  br i1 %v, label %loop_begin, label %loop_exit
; CHECK:       inner_loop_exit:
; CHECK-NEXT:    %[[A_INNER_LCSSA:.*]] = phi i32 [ %[[A]], %inner_loop_begin ]
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit

loop_exit:
  %a.lcssa = phi i32 [ %a.inner_lcssa, %inner_loop_exit ]
  ret i32 %a.lcssa
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A_INNER_LCSSA]], %inner_loop_exit ]
; CHECK-NEXT:    ret i32 %[[A_LCSSA]]
}

; The same core pattern as @test9a, but instead of the cloned loop becoming an
; infinite loop, the original loop has its only exit unswitched and the
; original loop becomes infinite and must be hoisted out of the loop nest.
define i32 @test9b(i1* %ptr, i1* %cond.ptr, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test9b(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %b = load i32, i32* %b.ptr
  %cond = load i1, i1* %cond.ptr
  br label %inner_loop_begin
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    %[[COND:.*]] = load i1, i1* %cond.ptr
; CHECK-NEXT:    br i1 %[[COND]], label %loop_begin.split.us, label %loop_begin.split

inner_loop_begin:
  %a = load i32, i32* %a.ptr
  br i1 %cond, label %inner_loop_exit, label %inner_loop_latch

inner_loop_latch:
  call void @sink1(i32 %b)
  br label %inner_loop_begin
; The cloned inner loop becomes a boring non-loop.
;
; CHECK:       loop_begin.split.us
; CHECK-NEXT:    br label %inner_loop_begin.us
;
; CHECK:       inner_loop_begin.us:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_exit.split.us
;
; CHECK:       inner_loop_exit.split.us
; CHECK-NEXT:    %[[A_INNER_LCSSA_US:.*]] = phi i32 [ %[[A]], %inner_loop_begin.us ]
; CHECK-NEXT:    br label %inner_loop_exit
;
; The original loop becomes an infinite loop and thus a top-level loop with the
; preheader as an exit block for the outer loop.
;
; CHECK:       loop_begin.split
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %loop_begin ]
; CHECK-NEXT:    br label %inner_loop_begin
;
; CHECK:       inner_loop_begin:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_latch
;
; CHECK:       inner_loop_latch:
; CHECK-NEXT:    call void @sink1(i32 %[[B_LCSSA]])
; CHECK-NEXT:    br label %inner_loop_begin

inner_loop_exit:
  %a.inner_lcssa = phi i32 [ %a, %inner_loop_begin ]
  %v = load i1, i1* %ptr
  br i1 %v, label %loop_begin, label %loop_exit
; CHECK:       inner_loop_exit:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit

loop_exit:
  %a.lcssa = phi i32 [ %a.inner_lcssa, %inner_loop_exit ]
  ret i32 %a.lcssa
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A_INNER_LCSSA_US]], %inner_loop_exit ]
; CHECK-NEXT:    ret i32 %[[A_LCSSA]]
}

; Test that requires re-forming dedicated exits for the cloned loop.
define i32 @test10a(i1* %ptr, i1 %cond, i32* %a.ptr) {
; CHECK-LABEL: @test10a(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond, label %entry.split.us, label %entry.split

loop_begin:
  %a = load i32, i32* %a.ptr
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %loop_a, label %loop_b

loop_a:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %loop_exit, label %loop_begin

loop_b:
  br i1 %cond, label %loop_exit, label %loop_begin
; The cloned loop with one edge as a direct exit.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_a.us, label %loop_b.us
;
; CHECK:       loop_b.us:
; CHECK-NEXT:    %[[A_LCSSA_B:.*]] = phi i32 [ %[[A]], %loop_begin.us ]
; CHECK-NEXT:    br label %loop_exit.split.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_exit.split.us.loopexit, label %loop_begin.backedge.us
;
; CHECK:       loop_begin.backedge.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_exit.split.us.loopexit:
; CHECK-NEXT:    %[[A_LCSSA_A:.*]] = phi i32 [ %[[A]], %loop_a.us ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    %[[A_PHI_US:.*]] = phi i32 [ %[[A_LCSSA_B]], %loop_b.us ], [ %[[A_LCSSA_A]], %loop_exit.split.us.loopexit ]
; CHECK-NEXT:    br label %loop_exit

; The original loop without one 'loop_exit' edge.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_a, label %loop_b
;
; CHECK:       loop_a:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_exit.split, label %loop_begin.backedge
;
; CHECK:       loop_begin.backedge:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_b:
; CHECK-NEXT:    br label %loop_begin.backedge
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A]], %loop_a ]
; CHECK-NEXT:    br label %loop_exit

loop_exit:
  %a.lcssa = phi i32 [ %a, %loop_a ], [ %a, %loop_b ]
  ret i32 %a.lcssa
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_PHI:.*]] = phi i32 [ %[[A_LCSSA]], %loop_exit.split ], [ %[[A_PHI_US]], %loop_exit.split.us ]
; CHECK-NEXT:    ret i32 %[[A_PHI]]
}

; Test that requires re-forming dedicated exits for the original loop.
define i32 @test10b(i1* %ptr, i1 %cond, i32* %a.ptr) {
; CHECK-LABEL: @test10b(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond, label %entry.split.us, label %entry.split

loop_begin:
  %a = load i32, i32* %a.ptr
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %loop_a, label %loop_b

loop_a:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %loop_begin, label %loop_exit

loop_b:
  br i1 %cond, label %loop_begin, label %loop_exit
; The cloned loop without one of the exits.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_a.us, label %loop_b.us
;
; CHECK:       loop_b.us:
; CHECK-NEXT:    br label %loop_begin.backedge.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.backedge.us, label %loop_exit.split.us
;
; CHECK:       loop_begin.backedge.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    %[[A_LCSSA_US:.*]] = phi i32 [ %[[A]], %loop_a.us ]
; CHECK-NEXT:    br label %loop_exit

; The original loop without one 'loop_exit' edge.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_a, label %loop_b
;
; CHECK:       loop_a:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.backedge, label %loop_exit.split.loopexit
;
; CHECK:       loop_begin.backedge:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_b:
; CHECK-NEXT:    %[[A_LCSSA_B:.*]] = phi i32 [ %[[A]], %loop_begin ]
; CHECK-NEXT:    br label %loop_exit.split
;
; CHECK:       loop_exit.split.loopexit:
; CHECK-NEXT:    %[[A_LCSSA_A:.*]] = phi i32 [ %[[A]], %loop_a ]
; CHECK-NEXT:    br label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[A_PHI_SPLIT:.*]] = phi i32 [ %[[A_LCSSA_B]], %loop_b ], [ %[[A_LCSSA_A]], %loop_exit.split.loopexit ]
; CHECK-NEXT:    br label %loop_exit

loop_exit:
  %a.lcssa = phi i32 [ %a, %loop_a ], [ %a, %loop_b ]
  ret i32 %a.lcssa
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_PHI:.*]] = phi i32 [ %[[A_PHI_SPLIT]], %loop_exit.split ], [ %[[A_LCSSA_US]], %loop_exit.split.us ]
; CHECK-NEXT:    ret i32 %[[A_PHI]]
}

; Check that if a cloned inner loop after unswitching doesn't loop and directly
; exits even an outer loop, we don't add the cloned preheader to the outer
; loop and do add the needed LCSSA phi nodes for the new exit block from the
; outer loop.
define i32 @test11a(i1* %ptr, i1* %cond.ptr, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test11a(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %b = load i32, i32* %b.ptr
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %loop_latch, label %inner_loop_ph
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_latch, label %inner_loop_ph

inner_loop_ph:
  %cond = load i1, i1* %cond.ptr
  br label %inner_loop_begin
; CHECK:       inner_loop_ph:
; CHECK-NEXT:    %[[COND:.*]] = load i1, i1* %cond.ptr
; CHECK-NEXT:    br i1 %[[COND]], label %inner_loop_ph.split.us, label %inner_loop_ph.split

inner_loop_begin:
  call void @sink1(i32 %b)
  %a = load i32, i32* %a.ptr
  br i1 %cond, label %loop_exit, label %inner_loop_a

inner_loop_a:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %inner_loop_exit, label %inner_loop_begin
; The cloned path doesn't actually loop and is an exit from the outer loop as
; well.
;
; CHECK:       inner_loop_ph.split.us:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %inner_loop_ph ]
; CHECK-NEXT:    br label %inner_loop_begin.us
;
; CHECK:       inner_loop_begin.us:
; CHECK-NEXT:    call void @sink1(i32 %[[B_LCSSA]])
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_exit.loopexit.split.us
;
; CHECK:       loop_exit.loopexit.split.us:
; CHECK-NEXT:    %[[A_INNER_LCSSA_US:.*]] = phi i32 [ %[[A]], %inner_loop_begin.us ]
; CHECK-NEXT:    br label %loop_exit.loopexit
;
; The original remains a loop losing the exit edge.
;
; CHECK:       inner_loop_ph.split:
; CHECK-NEXT:    br label %inner_loop_begin
;
; CHECK:       inner_loop_begin:
; CHECK-NEXT:    call void @sink1(i32 %[[B]])
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_a
;
; CHECK:       inner_loop_a:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_exit, label %inner_loop_begin

inner_loop_exit:
  %a.inner_lcssa = phi i32 [ %a, %inner_loop_a ]
  %v3 = load i1, i1* %ptr
  br i1 %v3, label %loop_latch, label %loop_exit
; CHECK:       inner_loop_exit:
; CHECK-NEXT:    %[[A_INNER_LCSSA:.*]] = phi i32 [ %[[A]], %inner_loop_a ]
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_latch, label %loop_exit.loopexit1

loop_latch:
  br label %loop_begin
; CHECK:       loop_latch:
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  %a.lcssa = phi i32 [ %a, %inner_loop_begin ], [ %a.inner_lcssa, %inner_loop_exit ]
  ret i32 %a.lcssa
; CHECK:       loop_exit.loopexit:
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit.loopexit1:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A_INNER_LCSSA]], %inner_loop_exit ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_PHI:.*]] = phi i32 [ %[[A_INNER_LCSSA_US]], %loop_exit.loopexit ], [ %[[A_LCSSA]], %loop_exit.loopexit1 ]
; CHECK-NEXT:    ret i32 %[[A_PHI]]
}

; Check that if the original inner loop after unswitching doesn't loop and
; directly exits even an outer loop, we remove the original preheader from the
; outer loop and add needed LCSSA phi nodes for the new exit block from the
; outer loop.
define i32 @test11b(i1* %ptr, i1* %cond.ptr, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test11b(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %b = load i32, i32* %b.ptr
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %loop_latch, label %inner_loop_ph
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_latch, label %inner_loop_ph

inner_loop_ph:
  %cond = load i1, i1* %cond.ptr
  br label %inner_loop_begin
; CHECK:       inner_loop_ph:
; CHECK-NEXT:    %[[COND:.*]] = load i1, i1* %cond.ptr
; CHECK-NEXT:    br i1 %[[COND]], label %inner_loop_ph.split.us, label %inner_loop_ph.split

inner_loop_begin:
  call void @sink1(i32 %b)
  %a = load i32, i32* %a.ptr
  br i1 %cond, label %inner_loop_a, label %loop_exit

inner_loop_a:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %inner_loop_exit, label %inner_loop_begin
; The cloned path continues to loop without the exit out of the entire nest.
;
; CHECK:       inner_loop_ph.split.us:
; CHECK-NEXT:    br label %inner_loop_begin.us
;
; CHECK:       inner_loop_begin.us:
; CHECK-NEXT:    call void @sink1(i32 %[[B]])
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_a.us
;
; CHECK:       inner_loop_a.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_exit.split.us, label %inner_loop_begin.us
;
; CHECK:       inner_loop_exit.split.us:
; CHECK-NEXT:    %[[A_INNER_LCSSA_US:.*]] = phi i32 [ %[[A]], %inner_loop_a.us ]
; CHECK-NEXT:    br label %inner_loop_exit
;
; The original remains a loop losing the exit edge.
;
; CHECK:       inner_loop_ph.split:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %inner_loop_ph ]
; CHECK-NEXT:    br label %inner_loop_begin
;
; CHECK:       inner_loop_begin:
; CHECK-NEXT:    call void @sink1(i32 %[[B_LCSSA]])
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %loop_exit.loopexit

inner_loop_exit:
  %a.inner_lcssa = phi i32 [ %a, %inner_loop_a ]
  %v3 = load i1, i1* %ptr
  br i1 %v3, label %loop_latch, label %loop_exit
; CHECK:       inner_loop_exit:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_latch, label %loop_exit.loopexit1

loop_latch:
  br label %loop_begin
; CHECK:       loop_latch:
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  %a.lcssa = phi i32 [ %a, %inner_loop_begin ], [ %a.inner_lcssa, %inner_loop_exit ]
  ret i32 %a.lcssa
; CHECK:       loop_exit.loopexit:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A]], %inner_loop_begin ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit.loopexit1:
; CHECK-NEXT:    %[[A_LCSSA_US:.*]] = phi i32 [ %[[A_INNER_LCSSA_US]], %inner_loop_exit ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_PHI:.*]] = phi i32 [ %[[A_LCSSA]], %loop_exit.loopexit ], [ %[[A_LCSSA_US]], %loop_exit.loopexit1 ]
; CHECK-NEXT:    ret i32 %[[A_PHI]]
}

; Like test11a, but checking that when the whole thing is wrapped in yet
; another loop, we correctly attribute the cloned preheader to that outermost
; loop rather than only handling the case where the preheader is not in any loop
; at all.
define i32 @test12a(i1* %ptr, i1* %cond.ptr, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test12a(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  br label %inner_loop_begin
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %inner_loop_begin

inner_loop_begin:
  %b = load i32, i32* %b.ptr
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %inner_loop_latch, label %inner_inner_loop_ph
; CHECK:       inner_loop_begin:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_latch, label %inner_inner_loop_ph

inner_inner_loop_ph:
  %cond = load i1, i1* %cond.ptr
  br label %inner_inner_loop_begin
; CHECK:       inner_inner_loop_ph:
; CHECK-NEXT:    %[[COND:.*]] = load i1, i1* %cond.ptr
; CHECK-NEXT:    br i1 %[[COND]], label %inner_inner_loop_ph.split.us, label %inner_inner_loop_ph.split

inner_inner_loop_begin:
  call void @sink1(i32 %b)
  %a = load i32, i32* %a.ptr
  br i1 %cond, label %inner_loop_exit, label %inner_inner_loop_a

inner_inner_loop_a:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %inner_inner_loop_exit, label %inner_inner_loop_begin
; The cloned path doesn't actually loop and is an exit from the outer loop as
; well.
;
; CHECK:       inner_inner_loop_ph.split.us:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %inner_inner_loop_ph ]
; CHECK-NEXT:    br label %inner_inner_loop_begin.us
;
; CHECK:       inner_inner_loop_begin.us:
; CHECK-NEXT:    call void @sink1(i32 %[[B_LCSSA]])
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_exit.loopexit.split.us
;
; CHECK:       inner_loop_exit.loopexit.split.us:
; CHECK-NEXT:    %[[A_INNER_INNER_LCSSA_US:.*]] = phi i32 [ %[[A]], %inner_inner_loop_begin.us ]
; CHECK-NEXT:    br label %inner_loop_exit.loopexit
;
; The original remains a loop losing the exit edge.
;
; CHECK:       inner_inner_loop_ph.split:
; CHECK-NEXT:    br label %inner_inner_loop_begin
;
; CHECK:       inner_inner_loop_begin:
; CHECK-NEXT:    call void @sink1(i32 %[[B]])
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_inner_loop_a
;
; CHECK:       inner_inner_loop_a:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_exit, label %inner_inner_loop_begin

inner_inner_loop_exit:
  %a.inner_inner_lcssa = phi i32 [ %a, %inner_inner_loop_a ]
  %v3 = load i1, i1* %ptr
  br i1 %v3, label %inner_loop_latch, label %inner_loop_exit
; CHECK:       inner_inner_loop_exit:
; CHECK-NEXT:    %[[A_INNER_INNER_LCSSA:.*]] = phi i32 [ %[[A]], %inner_inner_loop_a ]
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_latch, label %inner_loop_exit.loopexit1

inner_loop_latch:
  br label %inner_loop_begin
; CHECK:       inner_loop_latch:
; CHECK-NEXT:    br label %inner_loop_begin

inner_loop_exit:
  %a.inner_lcssa = phi i32 [ %a, %inner_inner_loop_begin ], [ %a.inner_inner_lcssa, %inner_inner_loop_exit ]
  %v4 = load i1, i1* %ptr
  br i1 %v4, label %loop_begin, label %loop_exit
; CHECK:       inner_loop_exit.loopexit:
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit.loopexit1:
; CHECK-NEXT:    %[[A_INNER_LCSSA:.*]] = phi i32 [ %[[A_INNER_INNER_LCSSA]], %inner_inner_loop_exit ]
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit:
; CHECK-NEXT:    %[[A_INNER_PHI:.*]] = phi i32 [ %[[A_INNER_INNER_LCSSA_US]], %inner_loop_exit.loopexit ], [ %[[A_INNER_LCSSA]], %inner_loop_exit.loopexit1 ]
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit

loop_exit:
  %a.lcssa = phi i32 [ %a.inner_lcssa, %inner_loop_exit ]
  ret i32 %a.lcssa
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_loop_exit ]
; CHECK-NEXT:    ret i32 %[[A_LCSSA]]
}

; Like test11b, but checking that when the whole thing is wrapped in yet
; another loop, we correctly sink the preheader to the outermost loop rather
; than only handling the case where the preheader is completely removed from
; a loop.
define i32 @test12b(i1* %ptr, i1* %cond.ptr, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test12b(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  br label %inner_loop_begin
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %inner_loop_begin

inner_loop_begin:
  %b = load i32, i32* %b.ptr
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %inner_loop_latch, label %inner_inner_loop_ph
; CHECK:       inner_loop_begin:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_latch, label %inner_inner_loop_ph

inner_inner_loop_ph:
  %cond = load i1, i1* %cond.ptr
  br label %inner_inner_loop_begin
; CHECK:       inner_inner_loop_ph:
; CHECK-NEXT:    %[[COND:.*]] = load i1, i1* %cond.ptr
; CHECK-NEXT:    br i1 %[[COND]], label %inner_inner_loop_ph.split.us, label %inner_inner_loop_ph.split

inner_inner_loop_begin:
  call void @sink1(i32 %b)
  %a = load i32, i32* %a.ptr
  br i1 %cond, label %inner_inner_loop_a, label %inner_loop_exit

inner_inner_loop_a:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %inner_inner_loop_exit, label %inner_inner_loop_begin
; The cloned path continues to loop without the exit out of the entire nest.
;
; CHECK:       inner_inner_loop_ph.split.us:
; CHECK-NEXT:    br label %inner_inner_loop_begin.us
;
; CHECK:       inner_inner_loop_begin.us:
; CHECK-NEXT:    call void @sink1(i32 %[[B]])
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_inner_loop_a.us
;
; CHECK:       inner_inner_loop_a.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_inner_loop_exit.split.us, label %inner_inner_loop_begin.us
;
; CHECK:       inner_inner_loop_exit.split.us:
; CHECK-NEXT:    %[[A_INNER_INNER_LCSSA_US:.*]] = phi i32 [ %[[A]], %inner_inner_loop_a.us ]
; CHECK-NEXT:    br label %inner_inner_loop_exit
;
; The original remains a loop losing the exit edge.
;
; CHECK:       inner_inner_loop_ph.split:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %inner_inner_loop_ph ]
; CHECK-NEXT:    br label %inner_inner_loop_begin
;
; CHECK:       inner_inner_loop_begin:
; CHECK-NEXT:    call void @sink1(i32 %[[B_LCSSA]])
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    br label %inner_loop_exit.loopexit

inner_inner_loop_exit:
  %a.inner_inner_lcssa = phi i32 [ %a, %inner_inner_loop_a ]
  %v3 = load i1, i1* %ptr
  br i1 %v3, label %inner_loop_latch, label %inner_loop_exit
; CHECK:       inner_inner_loop_exit:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %inner_loop_latch, label %inner_loop_exit.loopexit1

inner_loop_latch:
  br label %inner_loop_begin
; CHECK:       inner_loop_latch:
; CHECK-NEXT:    br label %inner_loop_begin

inner_loop_exit:
  %a.inner_lcssa = phi i32 [ %a, %inner_inner_loop_begin ], [ %a.inner_inner_lcssa, %inner_inner_loop_exit ]
  %v4 = load i1, i1* %ptr
  br i1 %v4, label %loop_begin, label %loop_exit
; CHECK:       inner_loop_exit.loopexit:
; CHECK-NEXT:    %[[A_INNER_LCSSA:.*]] = phi i32 [ %[[A]], %inner_inner_loop_begin ]
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit.loopexit1:
; CHECK-NEXT:    %[[A_INNER_LCSSA_US:.*]] = phi i32 [ %[[A_INNER_INNER_LCSSA_US]], %inner_inner_loop_exit ]
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit:
; CHECK-NEXT:    %[[A_INNER_PHI:.*]] = phi i32 [ %[[A_INNER_LCSSA]], %inner_loop_exit.loopexit ], [ %[[A_INNER_LCSSA_US]], %inner_loop_exit.loopexit1 ]
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit

loop_exit:
  %a.lcssa = phi i32 [ %a.inner_lcssa, %inner_loop_exit ]
  ret i32 %a.lcssa
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_loop_exit ]
; CHECK-NEXT:    ret i32 %[[A_LCSSA]]
}

; Test where the cloned loop has an inner loop that has to be traversed to form
; the cloned loop, and where this inner loop has multiple blocks, and where the
; exiting block that connects the inner loop to the cloned loop is not the header
; block. This ensures that we correctly handle interesting corner cases of
; traversing back to the header when establishing the cloned loop.
define i32 @test13a(i1* %ptr, i1 %cond, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test13a(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond, label %entry.split.us, label %entry.split

loop_begin:
  %a = load i32, i32* %a.ptr
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %loop_a, label %loop_b

loop_a:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %loop_exit, label %loop_latch

loop_b:
  %b = load i32, i32* %b.ptr
  br i1 %cond, label %loop_b_inner_ph, label %loop_exit

loop_b_inner_ph:
  br label %loop_b_inner_header

loop_b_inner_header:
  %v3 = load i1, i1* %ptr
  br i1 %v3, label %loop_b_inner_latch, label %loop_b_inner_body

loop_b_inner_body:
  %v4 = load i1, i1* %ptr
  br i1 %v4, label %loop_b_inner_latch, label %loop_b_inner_exit

loop_b_inner_latch:
  br label %loop_b_inner_header

loop_b_inner_exit:
  br label %loop_latch

loop_latch:
  br label %loop_begin
; The cloned loop contains an inner loop within it.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_a.us, label %loop_b.us
;
; CHECK:       loop_b.us:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br label %loop_b_inner_ph.us
;
; CHECK:       loop_b_inner_ph.us:
; CHECK-NEXT:    br label %loop_b_inner_header.us
;
; CHECK:       loop_b_inner_header.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_b_inner_latch.us, label %loop_b_inner_body.us
;
; CHECK:       loop_b_inner_body.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_b_inner_latch.us, label %loop_b_inner_exit.us
;
; CHECK:       loop_b_inner_exit.us:
; CHECK-NEXT:    br label %loop_latch.us
;
; CHECK:       loop_b_inner_latch.us:
; CHECK-NEXT:    br label %loop_b_inner_header.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_exit.split.us, label %loop_latch.us
;
; CHECK:       loop_latch.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    %[[A_LCSSA_US:.*]] = phi i32 [ %[[A]], %loop_a.us ]
; CHECK-NEXT:    br label %loop_exit
;
; And the original loop no longer contains an inner loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_a, label %loop_b
;
; CHECK:       loop_a:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_exit.split.loopexit, label %loop_latch
;
; CHECK:       loop_b:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br label %loop_exit.split
;
; CHECK:       loop_latch:
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  %lcssa = phi i32 [ %a, %loop_a ], [ %b, %loop_b ]
  ret i32 %lcssa
; CHECK:       loop_exit.split.loopexit:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A]], %loop_a ]
; CHECK-NEXT:    br label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[AB_PHI:.*]] = phi i32 [ %[[B]], %loop_b ], [ %[[A_LCSSA]], %loop_exit.split.loopexit ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[AB_PHI_US:.*]] = phi i32 [ %[[AB_PHI]], %loop_exit.split ], [ %[[A_LCSSA_US]], %loop_exit.split.us ]
; CHECK-NEXT:    ret i32 %[[AB_PHI_US]]
}

; Test where the original loop has an inner loop that has to be traversed to
; rebuild the loop, and where this inner loop has multiple blocks, and where
; the exiting block that connects the inner loop to the original loop is not
; the header block. This ensures that we correctly handle interesting corner
; cases of traversing back to the header when re-establishing the original loop
; still exists after unswitching.
define i32 @test13b(i1* %ptr, i1 %cond, i32* %a.ptr, i32* %b.ptr) {
; CHECK-LABEL: @test13b(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond, label %entry.split.us, label %entry.split

loop_begin:
  %a = load i32, i32* %a.ptr
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %loop_a, label %loop_b

loop_a:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %loop_exit, label %loop_latch

loop_b:
  %b = load i32, i32* %b.ptr
  br i1 %cond, label %loop_exit, label %loop_b_inner_ph

loop_b_inner_ph:
  br label %loop_b_inner_header

loop_b_inner_header:
  %v3 = load i1, i1* %ptr
  br i1 %v3, label %loop_b_inner_latch, label %loop_b_inner_body

loop_b_inner_body:
  %v4 = load i1, i1* %ptr
  br i1 %v4, label %loop_b_inner_latch, label %loop_b_inner_exit

loop_b_inner_latch:
  br label %loop_b_inner_header

loop_b_inner_exit:
  br label %loop_latch

loop_latch:
  br label %loop_begin
; The cloned loop doesn't contain an inner loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_a.us, label %loop_b.us
;
; CHECK:       loop_b.us:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br label %loop_exit.split.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_exit.split.us.loopexit, label %loop_latch.us
;
; CHECK:       loop_latch.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_exit.split.us.loopexit:
; CHECK-NEXT:    %[[A_LCSSA_US:.*]] = phi i32 [ %[[A]], %loop_a.us ]
; CHECK-NEXT:    br label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    %[[AB_PHI_US:.*]] = phi i32 [ %[[B]], %loop_b.us ], [ %[[A_LCSSA_US]], %loop_exit.split.us.loopexit ]
; CHECK-NEXT:    br label %loop_exit
;
; But the original loop contains an inner loop that must be traversed.;
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[A:.*]] = load i32, i32* %a.ptr
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_a, label %loop_b
;
; CHECK:       loop_a:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_exit.split, label %loop_latch
;
; CHECK:       loop_b:
; CHECK-NEXT:    %[[B:.*]] = load i32, i32* %b.ptr
; CHECK-NEXT:    br label %loop_b_inner_ph
;
; CHECK:       loop_b_inner_ph:
; CHECK-NEXT:    br label %loop_b_inner_header
;
; CHECK:       loop_b_inner_header:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_b_inner_latch, label %loop_b_inner_body
;
; CHECK:       loop_b_inner_body:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_b_inner_latch, label %loop_b_inner_exit
;
; CHECK:       loop_b_inner_latch:
; CHECK-NEXT:    br label %loop_b_inner_header
;
; CHECK:       loop_b_inner_exit:
; CHECK-NEXT:    br label %loop_latch
;
; CHECK:       loop_latch:
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  %lcssa = phi i32 [ %a, %loop_a ], [ %b, %loop_b ]
  ret i32 %lcssa
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A]], %loop_a ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[AB_PHI:.*]] = phi i32 [ %[[A_LCSSA]], %loop_exit.split ], [ %[[AB_PHI_US]], %loop_exit.split.us ]
; CHECK-NEXT:    ret i32 %[[AB_PHI]]
}

define i32 @test20(i32* %var, i32 %cond1, i32 %cond2) {
; CHECK-LABEL: @test20(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 %cond2, label %[[ENTRY_SPLIT_EXIT:.*]] [
; CHECK-NEXT:      i32 0, label %[[ENTRY_SPLIT_A:.*]]
; CHECK-NEXT:      i32 1, label %[[ENTRY_SPLIT_A]]
; CHECK-NEXT:      i32 13, label %[[ENTRY_SPLIT_B:.*]]
; CHECK-NEXT:      i32 2, label %[[ENTRY_SPLIT_A]]
; CHECK-NEXT:      i32 42, label %[[ENTRY_SPLIT_C:.*]]
; CHECK-NEXT:    ]

loop_begin:
  %var_val = load i32, i32* %var
  switch i32 %cond2, label %loop_exit [
    i32 0, label %loop_a
    i32 1, label %loop_a
    i32 13, label %loop_b
    i32 2, label %loop_a
    i32 42, label %loop_c
  ]

loop_a:
  call i32 @a()
  br label %loop_latch
; Unswitched 'a' loop.
;
; CHECK:       [[ENTRY_SPLIT_A]]:
; CHECK-NEXT:    br label %[[LOOP_BEGIN_A:.*]]
;
; CHECK:       [[LOOP_BEGIN_A]]:
; CHECK-NEXT:    %{{.*}} = load i32, i32* %var
; CHECK-NEXT:    br label %[[LOOP_A:.*]]
;
; CHECK:       [[LOOP_A]]:
; CHECK-NEXT:    call i32 @a()
; CHECK-NEXT:    br label %[[LOOP_LATCH_A:.*]]
;
; CHECK:       [[LOOP_LATCH_A]]:
; CHECK:         br label %[[LOOP_BEGIN_A]]

loop_b:
  call i32 @b()
  br label %loop_latch
; Unswitched 'b' loop.
;
; CHECK:       [[ENTRY_SPLIT_B]]:
; CHECK-NEXT:    br label %[[LOOP_BEGIN_B:.*]]
;
; CHECK:       [[LOOP_BEGIN_B]]:
; CHECK-NEXT:    %{{.*}} = load i32, i32* %var
; CHECK-NEXT:    br label %[[LOOP_B:.*]]
;
; CHECK:       [[LOOP_B]]:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %[[LOOP_LATCH_B:.*]]
;
; CHECK:       [[LOOP_LATCH_B]]:
; CHECK:         br label %[[LOOP_BEGIN_B]]

loop_c:
  call i32 @c() noreturn nounwind
  br label %loop_latch
; Unswitched 'c' loop.
;
; CHECK:       [[ENTRY_SPLIT_C]]:
; CHECK-NEXT:    br label %[[LOOP_BEGIN_C:.*]]
;
; CHECK:       [[LOOP_BEGIN_C]]:
; CHECK-NEXT:    %{{.*}} = load i32, i32* %var
; CHECK-NEXT:    br label %[[LOOP_C:.*]]
;
; CHECK:       [[LOOP_C]]:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %[[LOOP_LATCH_C:.*]]
;
; CHECK:       [[LOOP_LATCH_C]]:
; CHECK:         br label %[[LOOP_BEGIN_C]]

loop_latch:
  br label %loop_begin

loop_exit:
  %lcssa = phi i32 [ %var_val, %loop_begin ]
  ret i32 %lcssa
; Unswitched exit edge (no longer a loop).
;
; CHECK:       [[ENTRY_SPLIT_EXIT]]:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i32, i32* %var
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[LCSSA:.*]] = phi i32 [ %[[V]], %loop_begin ]
; CHECK-NEXT:    ret i32 %[[LCSSA]]
}

; Negative test: we do not switch when the loop contains unstructured control
; flows as it would significantly complicate the process as novel loops might
; be formed, etc.
define void @test_no_unswitch_unstructured_cfg(i1* %ptr, i1 %cond) {
; CHECK-LABEL: @test_no_unswitch_unstructured_cfg(
entry:
  br label %loop_begin

loop_begin:
  br i1 %cond, label %loop_left, label %loop_right

loop_left:
  %v1 = load i1, i1* %ptr
  br i1 %v1, label %loop_right, label %loop_merge

loop_right:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %loop_left, label %loop_merge

loop_merge:
  %v3 = load i1, i1* %ptr
  br i1 %v3, label %loop_latch, label %loop_exit

loop_latch:
  br label %loop_begin

loop_exit:
  ret void
}

; A test reduced out of 403.gcc with interesting nested loops that trigger
; multiple unswitches. A key component of this test is that there are multiple
; paths to reach an inner loop after unswitching, and one of them is via the
; predecessors of the unswitched loop header. That can allow us to find the loop
; through multiple different paths.
define void @test21(i1 %a, i1 %b) {
; CHECK-LABEL: @test21(
bb:
  br label %bb3
; CHECK-NOT:     br i1 %a
;
; CHECK:         br i1 %a, label %[[BB_SPLIT_US:.*]], label %[[BB_SPLIT:.*]]
;
; CHECK-NOT:     br i1 %a
; CHECK-NOT:     br i1 %b
;
; CHECK:       [[BB_SPLIT]]:
; CHECK:         br i1 %b
;
; CHECK-NOT:     br i1 %a
; CHECK-NOT:     br i1 %b

bb3:
  %tmp1.0 = phi i32 [ 0, %bb ], [ %tmp1.3, %bb23 ]
  br label %bb7

bb7:
  %tmp.0 = phi i1 [ true, %bb3 ], [ false, %bb19 ]
  %tmp1.1 = phi i32 [ %tmp1.0, %bb3 ], [ %tmp1.2.lcssa, %bb19 ]
  br i1 %tmp.0, label %bb11.preheader, label %bb23

bb11.preheader:
  br i1 %a, label %bb19, label %bb14.lr.ph

bb14.lr.ph:
  br label %bb14

bb14:
  %tmp2.02 = phi i32 [ 0, %bb14.lr.ph ], [ 1, %bb14 ]
  br i1 %b, label %bb11.bb19_crit_edge, label %bb14

bb11.bb19_crit_edge:
  %split = phi i32 [ %tmp2.02, %bb14 ]
  br label %bb19

bb19:
  %tmp1.2.lcssa = phi i32 [ %split, %bb11.bb19_crit_edge ], [ %tmp1.1, %bb11.preheader ]
  %tmp21 = icmp eq i32 %tmp1.2.lcssa, 0
  br i1 %tmp21, label %bb23, label %bb7

bb23:
  %tmp1.3 = phi i32 [ %tmp1.2.lcssa, %bb19 ], [ %tmp1.1, %bb7 ]
  br label %bb3
}

; A test reduced out of 400.perlbench that when unswitching the `%stop`
; condition clones a loop nest outside of a containing loop. This excercises a
; different cloning path from our other test cases and in turn verifying the
; resulting structure can catch any failures to correctly clone these nested
; loops.
declare void @f()
declare void @g()
declare i32 @h(i32 %arg)
define void @test22(i32 %arg) {
; CHECK-LABEL: define void @test22(
entry:
  br label %loop1.header

loop1.header:
  %stop = phi i1 [ true, %loop1.latch ], [ false, %entry ]
  %i = phi i32 [ %i.lcssa, %loop1.latch ], [ %arg, %entry ]
; CHECK:         %[[I:.*]] = phi i32 [ %{{.*}}, %loop1.latch ], [ %arg, %entry ]
  br i1 %stop, label %loop1.exit, label %loop1.body.loop2.ph
; CHECK:         br i1 %stop, label %loop1.exit, label %loop1.body.loop2.ph

loop1.body.loop2.ph:
  br label %loop2.header
; Just check that the we unswitched the key condition and that leads to the
; inner loop header.
;
; CHECK:       loop1.body.loop2.ph:
; CHECK-NEXT:    br i1 %stop, label %[[SPLIT_US:.*]], label %[[SPLIT:.*]]
;
; CHECK:       [[SPLIT_US]]:
; CHECK-NEXT:    br label %[[LOOP2_HEADER_US:.*]]
;
; CHECK:       [[LOOP2_HEADER_US]]:
; CHECK-NEXT:    %{{.*}} = phi i32 [ %[[I]], %[[SPLIT_US]] ]
;
; CHECK:       [[SPLIT]]:
; CHECK-NEXT:    br label %[[LOOP2_HEADER:.*]]
;
; CHECK:       [[LOOP2_HEADER]]:
; CHECK-NEXT:    %{{.*}} = phi i32 [ %[[I]], %[[SPLIT]] ]

loop2.header:
  %i.inner = phi i32 [ %i, %loop1.body.loop2.ph ], [ %i.next, %loop2.latch ]
  br label %loop3.header

loop3.header:
  %sw = call i32 @h(i32 %i.inner)
  switch i32 %sw, label %loop3.exit [
    i32 32, label %loop3.header
    i32 59, label %loop2.latch
    i32 36, label %loop1.latch
  ]

loop2.latch:
  %i.next = add i32 %i.inner, 1
  br i1 %stop, label %loop2.exit, label %loop2.header

loop1.latch:
  %i.lcssa = phi i32 [ %i.inner, %loop3.header ]
  br label %loop1.header

loop3.exit:
  call void @f()
  ret void

loop2.exit:
  call void @g()
  ret void

loop1.exit:
  call void @g()
  ret void
}

; Test that when we are unswitching and need to rebuild the loop block set we
; correctly skip past inner loops. We want to use the inner loop to efficiently
; skip whole subregions of the outer loop blocks but just because the header of
; the outer loop is also the preheader of an inner loop shouldn't confuse this
; walk.
define void @test23(i1 %arg, i1* %ptr) {
; CHECK-LABEL: define void @test23(
entry:
  br label %outer.header
; CHECK:       entry:
; CHECK-NEXT:    br i1 %arg,
;
; Just verify that we unswitched the correct bits. We should call `@f` twice in
; one unswitch and `@f` and then `@g` in the other.
; CHECK:         call void
; CHECK-SAME:              @f
; CHECK:         call void
; CHECK-SAME:              @f
;
; CHECK:         call void
; CHECK-SAME:              @f
; CHECK:         call void
; CHECK-SAME:              @g

outer.header:
  br label %inner.header

inner.header:
  call void @f()
  br label %inner.latch

inner.latch:
  %inner.cond = load i1, i1* %ptr
  br i1 %inner.cond, label %inner.header, label %outer.body

outer.body:
  br i1 %arg, label %outer.body.left, label %outer.body.right

outer.body.left:
  call void @f()
  br label %outer.latch

outer.body.right:
  call void @g()
  br label %outer.latch

outer.latch:
  %outer.cond = load i1, i1* %ptr
  br i1 %outer.cond, label %outer.header, label %exit

exit:
  ret void
}

; Non-trivial loop unswitching where there are two invariant conditions, but the
; second one is only in the cloned copy of the loop after unswitching.
define i32 @test24(i1* %ptr, i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @test24(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %entry.split.us, label %entry.split

loop_begin:
  br i1 %cond1, label %loop_a, label %loop_b

loop_a:
  br i1 %cond2, label %loop_a_a, label %loop_a_c
; The second unswitched condition.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br i1 %cond2, label %entry.split.us.split.us, label %entry.split.us.split

loop_a_a:
  call i32 @a()
  br label %latch
; The 'loop_a_a' unswitched loop.
;
; CHECK:       entry.split.us.split.us:
; CHECK-NEXT:    br label %loop_begin.us.us
;
; CHECK:       loop_begin.us.us:
; CHECK-NEXT:    br label %loop_a.us.us
;
; CHECK:       loop_a.us.us:
; CHECK-NEXT:    br label %loop_a_a.us.us
;
; CHECK:       loop_a_a.us.us:
; CHECK-NEXT:    call i32 @a()
; CHECK-NEXT:    br label %latch.us.us
;
; CHECK:       latch.us.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us.us, label %loop_exit.split.us.split.us
;
; CHECK:       loop_exit.split.us.split.us:
; CHECK-NEXT:    br label %loop_exit.split

loop_a_c:
  call i32 @c()
  br label %latch
; The 'loop_a_c' unswitched loop.
;
; CHECK:       entry.split.us.split:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    br label %loop_a.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    br label %loop_a_c.us
;
; CHECK:       loop_a_c.us:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %latch
;
; CHECK:       latch.us:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us, label %loop_exit.split.us.split
;
; CHECK:       loop_exit.split.us.split:
; CHECK-NEXT:    br label %loop_exit.split

loop_b:
  call i32 @b()
  br label %latch
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %loop_b
;
; CHECK:       loop_b:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %latch
;
; CHECK:       latch:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit

latch:
  %v = load i1, i1* %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret i32 0
; CHECK:       loop_exit:
; CHECK-NEXT:    ret
}

; Non-trivial partial loop unswitching of an invariant input to an 'or'.
define i32 @test25(i1* %ptr, i1 %cond) {
; CHECK-LABEL: @test25(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond, label %entry.split.us, label %entry.split

loop_begin:
  %v1 = load i1, i1* %ptr
  %cond_or = or i1 %v1, %cond
  br i1 %cond_or, label %loop_a, label %loop_b

loop_a:
  call i32 @a()
  br label %latch
; The 'loop_a' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    br label %loop_a.us
;
; CHECK:       loop_a.us:
; CHECK-NEXT:    call i32 @a()
; CHECK-NEXT:    br label %latch.us
;
; CHECK:       latch.us:
; CHECK-NEXT:    %[[V2_US:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V2_US]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    br label %loop_exit

loop_b:
  call i32 @b()
  br label %latch
; The original loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V1:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    %[[OR:.*]] = or i1 %[[V1]], false
; CHECK-NEXT:    br i1 %[[OR]], label %loop_a, label %loop_b
;
; CHECK:       loop_a:
; CHECK-NEXT:    call i32 @a()
; CHECK-NEXT:    br label %latch
;
; CHECK:       loop_b:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %latch

latch:
  %v2 = load i1, i1* %ptr
  br i1 %v2, label %loop_begin, label %loop_exit
; CHECK:       latch:
; CHECK-NEXT:    %[[V2:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V2]], label %loop_begin, label %loop_exit.split

loop_exit:
  ret i32 0
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    ret
}

; Non-trivial partial loop unswitching of multiple invariant inputs to an `and`
; chain.
define i32 @test26(i1* %ptr1, i1* %ptr2, i1* %ptr3, i1 %cond1, i1 %cond2, i1 %cond3) {
; CHECK-LABEL: @test26(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[INV_AND:.*]] = and i1 %cond3, %cond1
; CHECK-NEXT:    br i1 %[[INV_AND]], label %entry.split, label %entry.split.us

loop_begin:
  %v1 = load i1, i1* %ptr1
  %v2 = load i1, i1* %ptr2
  %cond_and1 = and i1 %v1, %cond1
  %cond_or1 = or i1 %v2, %cond2
  %cond_and2 = and i1 %cond_and1, %cond_or1
  %cond_and3 = and i1 %cond_and2, %cond3
  br i1 %cond_and3, label %loop_a, label %loop_b
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    br label %loop_b.us
;
; CHECK:       loop_b.us:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %latch.us
;
; CHECK:       latch.us:
; CHECK-NEXT:    %[[V3_US:.*]] = load i1, i1* %ptr3
; CHECK-NEXT:    br i1 %[[V3_US]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    br label %loop_exit

; The original loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V1:.*]] = load i1, i1* %ptr1
; CHECK-NEXT:    %[[V2:.*]] = load i1, i1* %ptr2
; CHECK-NEXT:    %[[AND1:.*]] = and i1 %[[V1]], true
; CHECK-NEXT:    %[[OR1:.*]] = or i1 %[[V2]], %cond2
; CHECK-NEXT:    %[[AND2:.*]] = and i1 %[[AND1]], %[[OR1]]
; CHECK-NEXT:    %[[AND3:.*]] = and i1 %[[AND2]], true
; CHECK-NEXT:    br i1 %[[AND3]], label %loop_a, label %loop_b

loop_a:
  call i32 @a()
  br label %latch
; CHECK:       loop_a:
; CHECK-NEXT:    call i32 @a()
; CHECK-NEXT:    br label %latch

loop_b:
  call i32 @b()
  br label %latch
; CHECK:       loop_b:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %latch

latch:
  %v3 = load i1, i1* %ptr3
  br i1 %v3, label %loop_begin, label %loop_exit
; CHECK:       latch:
; CHECK-NEXT:    %[[V3:.*]] = load i1, i1* %ptr3
; CHECK-NEXT:    br i1 %[[V3]], label %loop_begin, label %loop_exit.split

loop_exit:
  ret i32 0
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    ret
}

; Non-trivial partial loop unswitching of multiple invariant inputs to an `or`
; chain. Basically an inverted version of corresponding `and` test (test26).
define i32 @test27(i1* %ptr1, i1* %ptr2, i1* %ptr3, i1 %cond1, i1 %cond2, i1 %cond3) {
; CHECK-LABEL: @test27(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[INV_OR:.*]] = or i1 %cond3, %cond1
; CHECK-NEXT:    br i1 %[[INV_OR]], label %entry.split.us, label %entry.split

loop_begin:
  %v1 = load i1, i1* %ptr1
  %v2 = load i1, i1* %ptr2
  %cond_or1 = or i1 %v1, %cond1
  %cond_and1 = and i1 %v2, %cond2
  %cond_or2 = or i1 %cond_or1, %cond_and1
  %cond_or3 = or i1 %cond_or2, %cond3
  br i1 %cond_or3, label %loop_b, label %loop_a
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    br label %loop_b.us
;
; CHECK:       loop_b.us:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %latch.us
;
; CHECK:       latch.us:
; CHECK-NEXT:    %[[V3_US:.*]] = load i1, i1* %ptr3
; CHECK-NEXT:    br i1 %[[V3_US]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    br label %loop_exit

; The original loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V1:.*]] = load i1, i1* %ptr1
; CHECK-NEXT:    %[[V2:.*]] = load i1, i1* %ptr2
; CHECK-NEXT:    %[[OR1:.*]] = or i1 %[[V1]], false
; CHECK-NEXT:    %[[AND1:.*]] = and i1 %[[V2]], %cond2
; CHECK-NEXT:    %[[OR2:.*]] = or i1 %[[OR1]], %[[AND1]]
; CHECK-NEXT:    %[[OR3:.*]] = or i1 %[[OR2]], false
; CHECK-NEXT:    br i1 %[[OR3]], label %loop_b, label %loop_a

loop_a:
  call i32 @a()
  br label %latch
; CHECK:       loop_a:
; CHECK-NEXT:    call i32 @a()
; CHECK-NEXT:    br label %latch

loop_b:
  call i32 @b()
  br label %latch
; CHECK:       loop_b:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %latch

latch:
  %v3 = load i1, i1* %ptr3
  br i1 %v3, label %loop_begin, label %loop_exit
; CHECK:       latch:
; CHECK-NEXT:    %[[V3:.*]] = load i1, i1* %ptr3
; CHECK-NEXT:    br i1 %[[V3]], label %loop_begin, label %loop_exit.split

loop_exit:
  ret i32 0
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    ret
}

; Non-trivial unswitching of a switch.
define i32 @test28(i1* %ptr, i32 %cond) {
; CHECK-LABEL: @test28(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 %cond, label %[[ENTRY_SPLIT_LATCH:.*]] [
; CHECK-NEXT:      i32 0, label %[[ENTRY_SPLIT_A:.*]]
; CHECK-NEXT:      i32 1, label %[[ENTRY_SPLIT_B:.*]]
; CHECK-NEXT:      i32 2, label %[[ENTRY_SPLIT_C:.*]]
; CHECK-NEXT:    ]

loop_begin:
  switch i32 %cond, label %latch [
    i32 0, label %loop_a
    i32 1, label %loop_b
    i32 2, label %loop_c
  ]

loop_a:
  call i32 @a()
  br label %latch
; Unswitched 'a' loop.
;
; CHECK:       [[ENTRY_SPLIT_A]]:
; CHECK-NEXT:    br label %[[LOOP_BEGIN_A:.*]]
;
; CHECK:       [[LOOP_BEGIN_A]]:
; CHECK-NEXT:    br label %[[LOOP_A:.*]]
;
; CHECK:       [[LOOP_A]]:
; CHECK-NEXT:    call i32 @a()
; CHECK-NEXT:    br label %[[LOOP_LATCH_A:.*]]
;
; CHECK:       [[LOOP_LATCH_A]]:
; CHECK-NEXT:    %[[V_A:.*]] = load i1, i1* %ptr
; CHECK:         br i1 %[[V_A]], label %[[LOOP_BEGIN_A]], label %[[LOOP_EXIT_A:.*]]
;
; CHECK:       [[LOOP_EXIT_A]]:
; CHECK-NEXT:    br label %loop_exit

loop_b:
  call i32 @b()
  br label %latch
; Unswitched 'b' loop.
;
; CHECK:       [[ENTRY_SPLIT_B]]:
; CHECK-NEXT:    br label %[[LOOP_BEGIN_B:.*]]
;
; CHECK:       [[LOOP_BEGIN_B]]:
; CHECK-NEXT:    br label %[[LOOP_B:.*]]
;
; CHECK:       [[LOOP_B]]:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %[[LOOP_LATCH_B:.*]]
;
; CHECK:       [[LOOP_LATCH_B]]:
; CHECK-NEXT:    %[[V_B:.*]] = load i1, i1* %ptr
; CHECK:         br i1 %[[V_B]], label %[[LOOP_BEGIN_B]], label %[[LOOP_EXIT_B:.*]]
;
; CHECK:       [[LOOP_EXIT_B]]:
; CHECK-NEXT:    br label %loop_exit

loop_c:
  call i32 @c()
  br label %latch
; Unswitched 'c' loop.
;
; CHECK:       [[ENTRY_SPLIT_C]]:
; CHECK-NEXT:    br label %[[LOOP_BEGIN_C:.*]]
;
; CHECK:       [[LOOP_BEGIN_C]]:
; CHECK-NEXT:    br label %[[LOOP_C:.*]]
;
; CHECK:       [[LOOP_C]]:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %[[LOOP_LATCH_C:.*]]
;
; CHECK:       [[LOOP_LATCH_C]]:
; CHECK-NEXT:    %[[V_C:.*]] = load i1, i1* %ptr
; CHECK:         br i1 %[[V_C]], label %[[LOOP_BEGIN_C]], label %[[LOOP_EXIT_C:.*]]
;
; CHECK:       [[LOOP_EXIT_C]]:
; CHECK-NEXT:    br label %loop_exit

latch:
  %v = load i1, i1* %ptr
  br i1 %v, label %loop_begin, label %loop_exit
; Unswitched the 'latch' only loop.
;
; CHECK:       [[ENTRY_SPLIT_LATCH]]:
; CHECK-NEXT:    br label %[[LOOP_BEGIN_LATCH:.*]]
;
; CHECK:       [[LOOP_BEGIN_LATCH]]:
; CHECK-NEXT:    br label %[[LOOP_LATCH_LATCH:.*]]
;
; CHECK:       [[LOOP_LATCH_LATCH]]:
; CHECK-NEXT:    %[[V_LATCH:.*]] = load i1, i1* %ptr
; CHECK:         br i1 %[[V_LATCH]], label %[[LOOP_BEGIN_LATCH]], label %[[LOOP_EXIT_LATCH:.*]]
;
; CHECK:       [[LOOP_EXIT_LATCH]]:
; CHECK-NEXT:    br label %loop_exit

loop_exit:
  ret i32 0
; CHECK:       loop_exit:
; CHECK-NEXT:    ret i32 0
}

; A test case designed to exercise unusual properties of switches: they
; can introduce multiple edges to successors. These need lots of special case
; handling as they get collapsed in many cases (domtree, the unswitch itself)
; but not in all cases (the PHI node operands).
define i32 @test29(i32 %arg) {
; CHECK-LABEL: @test29(
entry:
  br label %header
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 %arg, label %[[ENTRY_SPLIT_C:.*]] [
; CHECK-NEXT:      i32 0, label %[[ENTRY_SPLIT_A:.*]]
; CHECK-NEXT:      i32 1, label %[[ENTRY_SPLIT_A]]
; CHECK-NEXT:      i32 2, label %[[ENTRY_SPLIT_B:.*]]
; CHECK-NEXT:      i32 3, label %[[ENTRY_SPLIT_C]]
; CHECK-NEXT:    ]

header:
  %tmp = call i32 @d()
  %cmp1 = icmp eq i32 %tmp, 0
  ; We set up a chain through all the successors of the switch that doesn't
  ; involve the switch so that we can have interesting PHI nodes in them.
  br i1 %cmp1, label %body.a, label %dispatch

dispatch:
  ; Switch with multiple successors. We arrange the last successor to be the
  ; default to make the test case easier to read. This has a duplicate edge
  ; both to the default destination (which is completely superfluous but
  ; technically valid IR) and to a regular successor.
  switch i32 %arg, label %body.c [
    i32 0, label %body.a
    i32 1, label %body.a
    i32 2, label %body.b
    i32 3, label %body.c
  ]

body.a:
  %tmp.a.phi = phi i32 [ 0, %header ], [ %tmp, %dispatch ], [ %tmp, %dispatch ]
  %tmp.a = call i32 @a()
  %tmp.a.sum = add i32 %tmp.a.phi, %tmp.a
  br label %body.b
; Unswitched 'a' loop.
;
; CHECK:       [[ENTRY_SPLIT_A]]:
; CHECK-NEXT:    br label %[[HEADER_A:.*]]
;
; CHECK:       [[HEADER_A]]:
; CHECK-NEXT:    %[[TMP_A:.*]] = call i32 @d()
; CHECK-NEXT:    %[[CMP1_A:.*]] = icmp eq i32 %[[TMP_A]], 0
; CHECK-NEXT:    br i1 %[[CMP1_A]], label %[[BODY_A_A:.*]], label %[[DISPATCH_A:.*]]
;
; CHECK:       [[DISPATCH_A]]:
; CHECK-NEXT:    br label %[[BODY_A_A]]
;
; CHECK:       [[BODY_A_A]]:
; CHECK-NEXT:    %[[TMP_A_PHI_A:.*]] = phi i32 [ 0, %[[HEADER_A]] ], [ %[[TMP_A]], %[[DISPATCH_A]] ]
; CHECK-NEXT:    %[[TMP_A_A:.*]] = call i32 @a()
; CHECK-NEXT:    %[[TMP_A_SUM_A:.*]] = add i32 %[[TMP_A_PHI_A]], %[[TMP_A_A]]
; CHECK-NEXT:    br label %[[BODY_B_A:.*]]
;
; CHECK:       [[BODY_B_A]]:
; CHECK-NEXT:    %[[TMP_B_PHI_A:.*]] = phi i32 [ %[[TMP_A_SUM_A]], %[[BODY_A_A]] ]
; CHECK-NEXT:    %[[TMP_B_A:.*]] = call i32 @b()
; CHECK-NEXT:    %[[TMP_B_SUM_A:.*]] = add i32 %[[TMP_B_PHI_A]], %[[TMP_B_A]]
; CHECK-NEXT:    br label %[[BODY_C_A:.*]]
;
; CHECK:       [[BODY_C_A]]:
; CHECK-NEXT:    %[[TMP_C_PHI_A:.*]] = phi i32 [ %[[TMP_B_SUM_A]], %[[BODY_B_A]] ]
; CHECK-NEXT:    %[[TMP_C_A:.*]] = call i32 @c()
; CHECK-NEXT:    %[[TMP_C_SUM_A:.*]] = add i32 %[[TMP_C_PHI_A]], %[[TMP_C_A]]
; CHECK-NEXT:    br label %[[LATCH_A:.*]]
;
; CHECK:       [[LATCH_A]]:
; CHECK-NEXT:    %[[CMP2_A:.*]] = icmp slt i32 %[[TMP_C_SUM_A]], 42
; CHECK:         br i1 %[[CMP2_A]], label %[[HEADER_A]], label %[[LOOP_EXIT_A:.*]]
;
; CHECK:       [[LOOP_EXIT_A]]:
; CHECK-NEXT:    %[[LCSSA_A:.*]] = phi i32 [ %[[TMP_C_SUM_A]], %[[LATCH_A]] ]
; CHECK-NEXT:    br label %exit

body.b:
  %tmp.b.phi = phi i32 [ %tmp, %dispatch ], [ %tmp.a.sum, %body.a ]
  %tmp.b = call i32 @b()
  %tmp.b.sum = add i32 %tmp.b.phi, %tmp.b
  br label %body.c
; Unswitched 'b' loop.
;
; CHECK:       [[ENTRY_SPLIT_B]]:
; CHECK-NEXT:    br label %[[HEADER_B:.*]]
;
; CHECK:       [[HEADER_B]]:
; CHECK-NEXT:    %[[TMP_B:.*]] = call i32 @d()
; CHECK-NEXT:    %[[CMP1_B:.*]] = icmp eq i32 %[[TMP_B]], 0
; CHECK-NEXT:    br i1 %[[CMP1_B]], label %[[BODY_A_B:.*]], label %[[DISPATCH_B:.*]]
;
; CHECK:       [[DISPATCH_B]]:
; CHECK-NEXT:    br label %[[BODY_B_B:.*]]
;
; CHECK:       [[BODY_A_B]]:
; CHECK-NEXT:    %[[TMP_A_PHI_B:.*]] = phi i32 [ 0, %[[HEADER_B]] ]
; CHECK-NEXT:    %[[TMP_A_B:.*]] = call i32 @a()
; CHECK-NEXT:    %[[TMP_A_SUM_B:.*]] = add i32 %[[TMP_A_PHI_B]], %[[TMP_A_B]]
; CHECK-NEXT:    br label %[[BODY_B_B:.*]]
;
; CHECK:       [[BODY_B_B]]:
; CHECK-NEXT:    %[[TMP_B_PHI_B:.*]] = phi i32 [ %[[TMP_B]], %[[DISPATCH_B]] ], [ %[[TMP_A_SUM_B]], %[[BODY_A_B]] ]
; CHECK-NEXT:    %[[TMP_B_B:.*]] = call i32 @b()
; CHECK-NEXT:    %[[TMP_B_SUM_B:.*]] = add i32 %[[TMP_B_PHI_B]], %[[TMP_B_B]]
; CHECK-NEXT:    br label %[[BODY_C_B:.*]]
;
; CHECK:       [[BODY_C_B]]:
; CHECK-NEXT:    %[[TMP_C_PHI_B:.*]] = phi i32 [ %[[TMP_B_SUM_B]], %[[BODY_B_B]] ]
; CHECK-NEXT:    %[[TMP_C_B:.*]] = call i32 @c()
; CHECK-NEXT:    %[[TMP_C_SUM_B:.*]] = add i32 %[[TMP_C_PHI_B]], %[[TMP_C_B]]
; CHECK-NEXT:    br label %[[LATCH_B:.*]]
;
; CHECK:       [[LATCH_B]]:
; CHECK-NEXT:    %[[CMP2_B:.*]] = icmp slt i32 %[[TMP_C_SUM_B]], 42
; CHECK:         br i1 %[[CMP2_B]], label %[[HEADER_B]], label %[[LOOP_EXIT_B:.*]]
;
; CHECK:       [[LOOP_EXIT_B]]:
; CHECK-NEXT:    %[[LCSSA_B:.*]] = phi i32 [ %[[TMP_C_SUM_B]], %[[LATCH_B]] ]
; CHECK-NEXT:    br label %[[EXIT_SPLIT:.*]]

body.c:
  %tmp.c.phi = phi i32 [ %tmp, %dispatch ], [ %tmp, %dispatch ], [ %tmp.b.sum, %body.b ]
  %tmp.c = call i32 @c()
  %tmp.c.sum = add i32 %tmp.c.phi, %tmp.c
  br label %latch
; Unswitched 'c' loop.
;
; CHECK:       [[ENTRY_SPLIT_C]]:
; CHECK-NEXT:    br label %[[HEADER_C:.*]]
;
; CHECK:       [[HEADER_C]]:
; CHECK-NEXT:    %[[TMP_C:.*]] = call i32 @d()
; CHECK-NEXT:    %[[CMP1_C:.*]] = icmp eq i32 %[[TMP_C]], 0
; CHECK-NEXT:    br i1 %[[CMP1_C]], label %[[BODY_A_C:.*]], label %[[DISPATCH_C:.*]]
;
; CHECK:       [[DISPATCH_C]]:
; CHECK-NEXT:    br label %[[BODY_C_C:.*]]
;
; CHECK:       [[BODY_A_C]]:
; CHECK-NEXT:    %[[TMP_A_PHI_C:.*]] = phi i32 [ 0, %[[HEADER_C]] ]
; CHECK-NEXT:    %[[TMP_A_C:.*]] = call i32 @a()
; CHECK-NEXT:    %[[TMP_A_SUM_C:.*]] = add i32 %[[TMP_A_PHI_C]], %[[TMP_A_C]]
; CHECK-NEXT:    br label %[[BODY_B_C:.*]]
;
; CHECK:       [[BODY_B_C]]:
; CHECK-NEXT:    %[[TMP_B_PHI_C:.*]] = phi i32 [ %[[TMP_A_SUM_C]], %[[BODY_A_C]] ]
; CHECK-NEXT:    %[[TMP_B_C:.*]] = call i32 @b()
; CHECK-NEXT:    %[[TMP_B_SUM_C:.*]] = add i32 %[[TMP_B_PHI_C]], %[[TMP_B_C]]
; CHECK-NEXT:    br label %[[BODY_C_C:.*]]
;
; CHECK:       [[BODY_C_C]]:
; CHECK-NEXT:    %[[TMP_C_PHI_C:.*]] = phi i32 [ %[[TMP_C]], %[[DISPATCH_C]] ], [ %[[TMP_B_SUM_C]], %[[BODY_B_C]] ]
; CHECK-NEXT:    %[[TMP_C_C:.*]] = call i32 @c()
; CHECK-NEXT:    %[[TMP_C_SUM_C:.*]] = add i32 %[[TMP_C_PHI_C]], %[[TMP_C_C]]
; CHECK-NEXT:    br label %[[LATCH_C:.*]]
;
; CHECK:       [[LATCH_C]]:
; CHECK-NEXT:    %[[CMP2_C:.*]] = icmp slt i32 %[[TMP_C_SUM_C]], 42
; CHECK:         br i1 %[[CMP2_C]], label %[[HEADER_C]], label %[[LOOP_EXIT_C:.*]]
;
; CHECK:       [[LOOP_EXIT_C]]:
; CHECK-NEXT:    %[[LCSSA_C:.*]] = phi i32 [ %[[TMP_C_SUM_C]], %[[LATCH_C]] ]
; CHECK-NEXT:    br label %[[EXIT_SPLIT]]

latch:
  %cmp2 = icmp slt i32 %tmp.c.sum, 42
  br i1 %cmp2, label %header, label %exit

exit:
  %lcssa.phi = phi i32 [ %tmp.c.sum, %latch ]
  ret i32 %lcssa.phi
; CHECK:       [[EXIT_SPLIT]]:
; CHECK-NEXT:    %[[EXIT_PHI1:.*]] = phi i32 [ %[[LCSSA_C]], %[[LOOP_EXIT_C]] ], [ %[[LCSSA_B]], %[[LOOP_EXIT_B]] ]
; CHECK-NEXT:    br label %exit

; CHECK:       exit:
; CHECK-NEXT:    %[[EXIT_PHI2:.*]] = phi i32 [ %[[EXIT_PHI1]], %[[EXIT_SPLIT]] ], [ %[[LCSSA_A]], %[[LOOP_EXIT_A]] ]
; CHECK-NEXT:    ret i32 %[[EXIT_PHI2]]
}

; Similar to @test29 but designed to have one of the duplicate edges be
; a loop exit edge as those can in some cases be special. Among other things,
; this includes an LCSSA phi with multiple entries despite being a dedicated
; exit block.
define i32 @test30(i32 %arg) {
; CHECK-LABEL: define i32 @test30(
entry:
  br label %header
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 %arg, label %[[ENTRY_SPLIT_EXIT:.*]] [
; CHECK-NEXT:      i32 -1, label %[[ENTRY_SPLIT_EXIT]]
; CHECK-NEXT:      i32 0, label %[[ENTRY_SPLIT_A:.*]]
; CHECK-NEXT:      i32 1, label %[[ENTRY_SPLIT_B:.*]]
; CHECK-NEXT:      i32 2, label %[[ENTRY_SPLIT_B]]
; CHECK-NEXT:    ]

header:
  %tmp = call i32 @d()
  %cmp1 = icmp eq i32 %tmp, 0
  br i1 %cmp1, label %body.a, label %dispatch

dispatch:
  switch i32 %arg, label %loop.exit1 [
    i32 -1, label %loop.exit1
    i32 0, label %body.a
    i32 1, label %body.b
    i32 2, label %body.b
  ]

body.a:
  %tmp.a.phi = phi i32 [ 0, %header ], [ %tmp, %dispatch ]
  %tmp.a = call i32 @a()
  %tmp.a.sum = add i32 %tmp.a.phi, %tmp.a
  br label %body.b
; Unswitched 'a' loop.
;
; CHECK:       [[ENTRY_SPLIT_A]]:
; CHECK-NEXT:    br label %[[HEADER_A:.*]]
;
; CHECK:       [[HEADER_A]]:
; CHECK-NEXT:    %[[TMP_A:.*]] = call i32 @d()
; CHECK-NEXT:    %[[CMP1_A:.*]] = icmp eq i32 %[[TMP_A]], 0
; CHECK-NEXT:    br i1 %[[CMP1_A]], label %[[BODY_A_A:.*]], label %[[DISPATCH_A:.*]]
;
; CHECK:       [[DISPATCH_A]]:
; CHECK-NEXT:    br label %[[BODY_A_A]]
;
; CHECK:       [[BODY_A_A]]:
; CHECK-NEXT:    %[[TMP_A_PHI_A:.*]] = phi i32 [ 0, %[[HEADER_A]] ], [ %[[TMP_A]], %[[DISPATCH_A]] ]
; CHECK-NEXT:    %[[TMP_A_A:.*]] = call i32 @a()
; CHECK-NEXT:    %[[TMP_A_SUM_A:.*]] = add i32 %[[TMP_A_PHI_A]], %[[TMP_A_A]]
; CHECK-NEXT:    br label %[[BODY_B_A:.*]]
;
; CHECK:       [[BODY_B_A]]:
; CHECK-NEXT:    %[[TMP_B_PHI_A:.*]] = phi i32 [ %[[TMP_A_SUM_A]], %[[BODY_A_A]] ]
; CHECK-NEXT:    %[[TMP_B_A:.*]] = call i32 @b()
; CHECK-NEXT:    %[[TMP_B_SUM_A:.*]] = add i32 %[[TMP_B_PHI_A]], %[[TMP_B_A]]
; CHECK-NEXT:    br label %[[LATCH_A:.*]]
;
; CHECK:       [[LATCH_A]]:
; CHECK-NEXT:    %[[CMP2_A:.*]] = icmp slt i32 %[[TMP_B_SUM_A]], 42
; CHECK:         br i1 %[[CMP2_A]], label %[[HEADER_A]], label %[[LOOP_EXIT_A:.*]]
;
; CHECK:       [[LOOP_EXIT_A]]:
; CHECK-NEXT:    %[[LCSSA_A:.*]] = phi i32 [ %[[TMP_B_SUM_A]], %[[LATCH_A]] ]
; CHECK-NEXT:    br label %loop.exit2

body.b:
  %tmp.b.phi = phi i32 [ %tmp, %dispatch ], [ %tmp, %dispatch ], [ %tmp.a.sum, %body.a ]
  %tmp.b = call i32 @b()
  %tmp.b.sum = add i32 %tmp.b.phi, %tmp.b
  br label %latch
; Unswitched 'b' loop.
;
; CHECK:       [[ENTRY_SPLIT_B]]:
; CHECK-NEXT:    br label %[[HEADER_B:.*]]
;
; CHECK:       [[HEADER_B]]:
; CHECK-NEXT:    %[[TMP_B:.*]] = call i32 @d()
; CHECK-NEXT:    %[[CMP1_B:.*]] = icmp eq i32 %[[TMP_B]], 0
; CHECK-NEXT:    br i1 %[[CMP1_B]], label %[[BODY_A_B:.*]], label %[[DISPATCH_B:.*]]
;
; CHECK:       [[DISPATCH_B]]:
; CHECK-NEXT:    br label %[[BODY_B_B]]
;
; CHECK:       [[BODY_A_B]]:
; CHECK-NEXT:    %[[TMP_A_PHI_B:.*]] = phi i32 [ 0, %[[HEADER_B]] ]
; CHECK-NEXT:    %[[TMP_A_B:.*]] = call i32 @a()
; CHECK-NEXT:    %[[TMP_A_SUM_B:.*]] = add i32 %[[TMP_A_PHI_B]], %[[TMP_A_B]]
; CHECK-NEXT:    br label %[[BODY_B_B:.*]]
;
; CHECK:       [[BODY_B_B]]:
; CHECK-NEXT:    %[[TMP_B_PHI_B:.*]] = phi i32 [ %[[TMP_B]], %[[DISPATCH_B]] ], [ %[[TMP_A_SUM_B]], %[[BODY_A_B]] ]
; CHECK-NEXT:    %[[TMP_B_B:.*]] = call i32 @b()
; CHECK-NEXT:    %[[TMP_B_SUM_B:.*]] = add i32 %[[TMP_B_PHI_B]], %[[TMP_B_B]]
; CHECK-NEXT:    br label %[[LATCH_B:.*]]
;
; CHECK:       [[LATCH_B]]:
; CHECK-NEXT:    %[[CMP2_B:.*]] = icmp slt i32 %[[TMP_B_SUM_B]], 42
; CHECK:         br i1 %[[CMP2_B]], label %[[HEADER_B]], label %[[LOOP_EXIT_B:.*]]
;
; CHECK:       [[LOOP_EXIT_B]]:
; CHECK-NEXT:    %[[LCSSA_B:.*]] = phi i32 [ %[[TMP_B_SUM_B]], %[[LATCH_B]] ]
; CHECK-NEXT:    br label %[[LOOP_EXIT2_SPLIT:.*]]

latch:
  %cmp2 = icmp slt i32 %tmp.b.sum, 42
  br i1 %cmp2, label %header, label %loop.exit2

loop.exit1:
  %l1.phi = phi i32 [ %tmp, %dispatch ], [ %tmp, %dispatch ]
  br label %exit
; Unswitched 'exit' loop.
;
; CHECK:       [[ENTRY_SPLIT_EXIT]]:
; CHECK-NEXT:    br label %[[HEADER_EXIT:.*]]
;
; CHECK:       [[HEADER_EXIT]]:
; CHECK-NEXT:    %[[TMP_EXIT:.*]] = call i32 @d()
; CHECK-NEXT:    %[[CMP1_EXIT:.*]] = icmp eq i32 %[[TMP_EXIT]], 0
; CHECK-NEXT:    br i1 %[[CMP1_EXIT]], label %[[BODY_A_EXIT:.*]], label %[[DISPATCH_EXIT:.*]]
;
; CHECK:       [[DISPATCH_EXIT]]:
; CHECK-NEXT:    %[[TMP_LCSSA:.*]] = phi i32 [ %[[TMP_EXIT]], %[[HEADER_EXIT]] ]
; CHECK-NEXT:    br label %loop.exit1
;
; CHECK:       [[BODY_A_EXIT]]:
; CHECK-NEXT:    %[[TMP_A_PHI_EXIT:.*]] = phi i32 [ 0, %[[HEADER_EXIT]] ]
; CHECK-NEXT:    %[[TMP_A_EXIT:.*]] = call i32 @a()
; CHECK-NEXT:    %[[TMP_A_SUM_EXIT:.*]] = add i32 %[[TMP_A_PHI_EXIT]], %[[TMP_A_EXIT]]
; CHECK-NEXT:    br label %[[BODY_B_EXIT:.*]]
;
; CHECK:       [[BODY_B_EXIT]]:
; CHECK-NEXT:    %[[TMP_B_PHI_EXIT:.*]] = phi i32 [ %[[TMP_A_SUM_EXIT]], %[[BODY_A_EXIT]] ]
; CHECK-NEXT:    %[[TMP_B_EXIT:.*]] = call i32 @b()
; CHECK-NEXT:    %[[TMP_B_SUM_EXIT:.*]] = add i32 %[[TMP_B_PHI_EXIT]], %[[TMP_B_EXIT]]
; CHECK-NEXT:    br label %[[LATCH_EXIT:.*]]
;
; CHECK:       [[LATCH_EXIT]]:
; CHECK-NEXT:    %[[CMP2_EXIT:.*]] = icmp slt i32 %[[TMP_B_SUM_EXIT]], 42
; CHECK:         br i1 %[[CMP2_EXIT]], label %[[HEADER_EXIT]], label %[[LOOP_EXIT_EXIT:.*]]
;
; CHECK:       loop.exit1:
; CHECK-NEXT:    %[[L1_PHI:.*]] = phi i32 [ %[[TMP_LCSSA]], %[[DISPATCH_EXIT]] ]
; CHECK-NEXT:    br label %exit
;
; CHECK:       [[LOOP_EXIT_EXIT]]:
; CHECK-NEXT:    %[[L2_PHI:.*]] = phi i32 [ %[[TMP_B_SUM_EXIT]], %[[LATCH_EXIT]] ]
; CHECK-NEXT:    br label %[[LOOP_EXIT2_SPLIT]]

loop.exit2:
  %l2.phi = phi i32 [ %tmp.b.sum, %latch ]
  br label %exit
; CHECK:       [[LOOP_EXIT2_SPLIT]]:
; CHECK-NEXT:    %[[LOOP_EXIT_PHI1:.*]] = phi i32 [ %[[L2_PHI]], %[[LOOP_EXIT_EXIT]] ], [ %[[LCSSA_B]], %[[LOOP_EXIT_B]] ]
; CHECK-NEXT:    br label %loop.exit2
;
; CHECK:       loop.exit2:
; CHECK-NEXT:    %[[LOOP_EXIT_PHI2:.*]] = phi i32 [ %[[LOOP_EXIT_PHI1]], %[[LOOP_EXIT2_SPLIT]] ], [ %[[LCSSA_A]], %[[LOOP_EXIT_A]] ]
; CHECK-NEXT:    br label %exit

exit:
  %l.phi = phi i32 [ %l1.phi, %loop.exit1 ], [ %l2.phi, %loop.exit2 ]
  ret i32 %l.phi
; CHECK:       exit:
; CHECK-NEXT:    %[[EXIT_PHI:.*]] = phi i32 [ %[[L1_PHI]], %loop.exit1 ], [ %[[LOOP_EXIT_PHI2]], %loop.exit2 ]
; CHECK-NEXT:    ret i32 %[[EXIT_PHI]]
}

; Unswitch will not actually change the loop nest from:
;   A < B < C
define void @hoist_inner_loop0() {
; CHECK-LABEL: define void @hoist_inner_loop0(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    br label %b.header

b.header:
  %v1 = call i1 @cond()
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    %v1 = call i1 @cond()
; CHECK-NEXT:    br i1 %v1, label %[[B_HEADER_SPLIT_US:.*]], label %[[B_HEADER_SPLIT:.*]]
;
; CHECK:       [[B_HEADER_SPLIT_US]]:
; CHECK-NEXT:    br label %[[C_HEADER_US:.*]]
;
; CHECK:       [[C_HEADER_US]]:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %[[B_LATCH_SPLIT_US:.*]]
;
; CHECK:       [[B_LATCH_SPLIT_US]]:
; CHECK-NEXT:    br label %b.latch
;
; CHECK:       [[B_HEADER_SPLIT]]:
; CHECK-NEXT:    br label %c.header

c.header:
  call i32 @c()
  br i1 %v1, label %b.latch, label %c.latch
; CHECK:       c.header:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %c.latch

c.latch:
  %v2 = call i1 @cond()
  br i1 %v2, label %c.header, label %b.latch
; CHECK:       c.latch:
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %c.header, label %[[B_LATCH_SPLIT:.*]]

b.latch:
  %v3 = call i1 @cond()
  br i1 %v3, label %b.header, label %a.latch
; CHECK:       [[B_LATCH_SPLIT]]:
; CHECK-NEXT:    br label %b.latch
;
; CHECK:       b.latch:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %b.header, label %a.latch

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; Unswitch will transform the loop nest from:
;   A < B < C
; into
;   A < (B, C)
define void @hoist_inner_loop1(i32* %ptr) {
; CHECK-LABEL: define void @hoist_inner_loop1(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  %x.a = load i32, i32* %ptr
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    %x.a = load i32, i32* %ptr
; CHECK-NEXT:    br label %b.header

b.header:
  %x.b = load i32, i32* %ptr
  %v1 = call i1 @cond()
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    %x.b = load i32, i32* %ptr
; CHECK-NEXT:    %v1 = call i1 @cond()
; CHECK-NEXT:    br i1 %v1, label %[[B_HEADER_SPLIT_US:.*]], label %[[B_HEADER_SPLIT:.*]]
;
; CHECK:       [[B_HEADER_SPLIT_US]]:
; CHECK-NEXT:    br label %[[C_HEADER_US:.*]]
;
; CHECK:       [[C_HEADER_US]]:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %[[B_LATCH_US:.*]]
;
; CHECK:       [[B_LATCH_US]]:
; CHECK-NEXT:    br label %b.latch
;
; CHECK:       [[B_HEADER_SPLIT]]:
; CHECK-NEXT:    %[[X_B_LCSSA:.*]] = phi i32 [ %x.b, %b.header ]
; CHECK-NEXT:    br label %c.header

c.header:
  call i32 @c()
  br i1 %v1, label %b.latch, label %c.latch
; CHECK:       c.header:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %c.latch

c.latch:
  ; Use values from other loops to check LCSSA form.
  store i32 %x.a, i32* %ptr
  store i32 %x.b, i32* %ptr
  %v2 = call i1 @cond()
  br i1 %v2, label %c.header, label %a.exit.c
; CHECK:       c.latch:
; CHECK-NEXT:    store i32 %x.a, i32* %ptr
; CHECK-NEXT:    store i32 %[[X_B_LCSSA]], i32* %ptr
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %c.header, label %a.exit.c

b.latch:
  %v3 = call i1 @cond()
  br i1 %v3, label %b.header, label %a.exit.b
; CHECK:       b.latch:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %b.header, label %a.exit.b

a.exit.c:
  br label %a.latch
; CHECK:       a.exit.c
; CHECK-NEXT:    br label %a.latch

a.exit.b:
  br label %a.latch
; CHECK:       a.exit.b:
; CHECK-NEXT:    br label %a.latch

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; Unswitch will transform the loop nest from:
;   A < B < C
; into
;   (A < B), C
define void @hoist_inner_loop2(i32* %ptr) {
; CHECK-LABEL: define void @hoist_inner_loop2(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  %x.a = load i32, i32* %ptr
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    %x.a = load i32, i32* %ptr
; CHECK-NEXT:    br label %b.header

b.header:
  %x.b = load i32, i32* %ptr
  %v1 = call i1 @cond()
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    %x.b = load i32, i32* %ptr
; CHECK-NEXT:    %v1 = call i1 @cond()
; CHECK-NEXT:    br i1 %v1, label %[[B_HEADER_SPLIT_US:.*]], label %[[B_HEADER_SPLIT:.*]]
;
; CHECK:       [[B_HEADER_SPLIT_US]]:
; CHECK-NEXT:    br label %[[C_HEADER_US:.*]]
;
; CHECK:       [[C_HEADER_US]]:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %[[B_LATCH_US:.*]]
;
; CHECK:       [[B_LATCH_US]]:
; CHECK-NEXT:    br label %b.latch
;
; CHECK:       [[B_HEADER_SPLIT]]:
; CHECK-NEXT:    %[[X_A_LCSSA:.*]] = phi i32 [ %x.a, %b.header ]
; CHECK-NEXT:    %[[X_B_LCSSA:.*]] = phi i32 [ %x.b, %b.header ]
; CHECK-NEXT:    br label %c.header

c.header:
  call i32 @c()
  br i1 %v1, label %b.latch, label %c.latch
; CHECK:       c.header:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %c.latch

c.latch:
  ; Use values from other loops to check LCSSA form.
  store i32 %x.a, i32* %ptr
  store i32 %x.b, i32* %ptr
  %v2 = call i1 @cond()
  br i1 %v2, label %c.header, label %exit
; CHECK:       c.latch:
; CHECK-NEXT:    store i32 %[[X_A_LCSSA]], i32* %ptr
; CHECK-NEXT:    store i32 %[[X_B_LCSSA]], i32* %ptr
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %c.header, label %exit

b.latch:
  %v3 = call i1 @cond()
  br i1 %v3, label %b.header, label %a.latch
; CHECK:       b.latch:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %b.header, label %a.latch

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; Same as @hoist_inner_loop2 but with a nested loop inside the hoisted loop.
; Unswitch will transform the loop nest from:
;   A < B < C < D
; into
;   (A < B), (C < D)
define void @hoist_inner_loop3(i32* %ptr) {
; CHECK-LABEL: define void @hoist_inner_loop3(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  %x.a = load i32, i32* %ptr
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    %x.a = load i32, i32* %ptr
; CHECK-NEXT:    br label %b.header

b.header:
  %x.b = load i32, i32* %ptr
  %v1 = call i1 @cond()
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    %x.b = load i32, i32* %ptr
; CHECK-NEXT:    %v1 = call i1 @cond()
; CHECK-NEXT:    br i1 %v1, label %[[B_HEADER_SPLIT_US:.*]], label %[[B_HEADER_SPLIT:.*]]
;
; CHECK:       [[B_HEADER_SPLIT_US]]:
; CHECK-NEXT:    br label %[[C_HEADER_US:.*]]
;
; CHECK:       [[C_HEADER_US]]:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %[[B_LATCH_US:.*]]
;
; CHECK:       [[B_LATCH_US]]:
; CHECK-NEXT:    br label %b.latch
;
; CHECK:       [[B_HEADER_SPLIT]]:
; CHECK-NEXT:    %[[X_A_LCSSA:.*]] = phi i32 [ %x.a, %b.header ]
; CHECK-NEXT:    %[[X_B_LCSSA:.*]] = phi i32 [ %x.b, %b.header ]
; CHECK-NEXT:    br label %c.header

c.header:
  call i32 @c()
  br i1 %v1, label %b.latch, label %c.body
; CHECK:       c.header:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %c.body

c.body:
  %x.c = load i32, i32* %ptr
  br label %d.header
; CHECK:       c.body:
; CHECK-NEXT:    %x.c = load i32, i32* %ptr
; CHECK-NEXT:    br label %d.header

d.header:
  ; Use values from other loops to check LCSSA form.
  store i32 %x.a, i32* %ptr
  store i32 %x.b, i32* %ptr
  store i32 %x.c, i32* %ptr
  %v2 = call i1 @cond()
  br i1 %v2, label %d.header, label %c.latch
; CHECK:       d.header:
; CHECK-NEXT:    store i32 %[[X_A_LCSSA]], i32* %ptr
; CHECK-NEXT:    store i32 %[[X_B_LCSSA]], i32* %ptr
; CHECK-NEXT:    store i32 %x.c, i32* %ptr
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %d.header, label %c.latch

c.latch:
  %v3 = call i1 @cond()
  br i1 %v3, label %c.header, label %exit
; CHECK:       c.latch:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %c.header, label %exit

b.latch:
  %v4 = call i1 @cond()
  br i1 %v4, label %b.header, label %a.latch
; CHECK:       b.latch:
; CHECK-NEXT:    %v4 = call i1 @cond()
; CHECK-NEXT:    br i1 %v4, label %b.header, label %a.latch

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; This test is designed to exercise checking multiple remaining exits from the
; loop being unswitched.
; Unswitch will transform the loop nest from:
;   A < B < C < D
; into
;   A < B < (C, D)
define void @hoist_inner_loop4() {
; CHECK-LABEL: define void @hoist_inner_loop4(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    br label %b.header

b.header:
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    br label %c.header

c.header:
  %v1 = call i1 @cond()
  br label %d.header
; CHECK:       c.header:
; CHECK-NEXT:    %v1 = call i1 @cond()
; CHECK-NEXT:    br i1 %v1, label %[[C_HEADER_SPLIT_US:.*]], label %[[C_HEADER_SPLIT:.*]]
;
; CHECK:       [[C_HEADER_SPLIT_US]]:
; CHECK-NEXT:    br label %[[D_HEADER_US:.*]]
;
; CHECK:       [[D_HEADER_US]]:
; CHECK-NEXT:    call i32 @d()
; CHECK-NEXT:    br label %[[C_LATCH_US:.*]]
;
; CHECK:       [[C_LATCH_US]]:
; CHECK-NEXT:    br label %c.latch
;
; CHECK:       [[C_HEADER_SPLIT]]:
; CHECK-NEXT:    br label %d.header

d.header:
  call i32 @d()
  br i1 %v1, label %c.latch, label %d.exiting1
; CHECK:       d.header:
; CHECK-NEXT:    call i32 @d()
; CHECK-NEXT:    br label %d.exiting1

d.exiting1:
  %v2 = call i1 @cond()
  br i1 %v2, label %d.exiting2, label %a.latch
; CHECK:       d.exiting1:
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %d.exiting2, label %a.latch

d.exiting2:
  %v3 = call i1 @cond()
  br i1 %v3, label %d.exiting3, label %loopexit.d
; CHECK:       d.exiting2:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %d.exiting3, label %loopexit.d

d.exiting3:
  %v4 = call i1 @cond()
  br i1 %v4, label %d.latch, label %b.latch
; CHECK:       d.exiting3:
; CHECK-NEXT:    %v4 = call i1 @cond()
; CHECK-NEXT:    br i1 %v4, label %d.latch, label %b.latch

d.latch:
  br label %d.header
; CHECK:       d.latch:
; CHECK-NEXT:    br label %d.header

c.latch:
  %v5 = call i1 @cond()
  br i1 %v5, label %c.header, label %loopexit.c
; CHECK:       c.latch:
; CHECK-NEXT:    %v5 = call i1 @cond()
; CHECK-NEXT:    br i1 %v5, label %c.header, label %loopexit.c

b.latch:
  br label %b.header
; CHECK:       b.latch:
; CHECK-NEXT:    br label %b.header

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

loopexit.d:
  br label %exit
; CHECK:       loopexit.d:
; CHECK-NEXT:    br label %exit

loopexit.c:
  br label %exit
; CHECK:       loopexit.c:
; CHECK-NEXT:    br label %exit

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; Unswitch will transform the loop nest from:
;   A < B < C < D
; into
;   A < ((B < C), D)
define void @hoist_inner_loop5(i32* %ptr) {
; CHECK-LABEL: define void @hoist_inner_loop5(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  %x.a = load i32, i32* %ptr
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    %x.a = load i32, i32* %ptr
; CHECK-NEXT:    br label %b.header

b.header:
  %x.b = load i32, i32* %ptr
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    %x.b = load i32, i32* %ptr
; CHECK-NEXT:    br label %c.header

c.header:
  %x.c = load i32, i32* %ptr
  %v1 = call i1 @cond()
  br label %d.header
; CHECK:       c.header:
; CHECK-NEXT:    %x.c = load i32, i32* %ptr
; CHECK-NEXT:    %v1 = call i1 @cond()
; CHECK-NEXT:    br i1 %v1, label %[[C_HEADER_SPLIT_US:.*]], label %[[C_HEADER_SPLIT:.*]]
;
; CHECK:       [[C_HEADER_SPLIT_US]]:
; CHECK-NEXT:    br label %[[D_HEADER_US:.*]]
;
; CHECK:       [[D_HEADER_US]]:
; CHECK-NEXT:    call i32 @d()
; CHECK-NEXT:    br label %[[C_LATCH_US:.*]]
;
; CHECK:       [[C_LATCH_US]]:
; CHECK-NEXT:    br label %c.latch
;
; CHECK:       [[C_HEADER_SPLIT]]:
; CHECK-NEXT:    %[[X_B_LCSSA:.*]] = phi i32 [ %x.b, %c.header ]
; CHECK-NEXT:    %[[X_C_LCSSA:.*]] = phi i32 [ %x.c, %c.header ]
; CHECK-NEXT:    br label %d.header

d.header:
  call i32 @d()
  br i1 %v1, label %c.latch, label %d.latch
; CHECK:       d.header:
; CHECK-NEXT:    call i32 @d()
; CHECK-NEXT:    br label %d.latch

d.latch:
  ; Use values from other loops to check LCSSA form.
  store i32 %x.a, i32* %ptr
  store i32 %x.b, i32* %ptr
  store i32 %x.c, i32* %ptr
  %v2 = call i1 @cond()
  br i1 %v2, label %d.header, label %a.latch
; CHECK:       d.latch:
; CHECK-NEXT:    store i32 %x.a, i32* %ptr
; CHECK-NEXT:    store i32 %[[X_B_LCSSA]], i32* %ptr
; CHECK-NEXT:    store i32 %[[X_C_LCSSA]], i32* %ptr
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %d.header, label %a.latch

c.latch:
  %v3 = call i1 @cond()
  br i1 %v3, label %c.header, label %b.latch
; CHECK:       c.latch:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %c.header, label %b.latch

b.latch:
  br label %b.header
; CHECK:       b.latch:
; CHECK-NEXT:    br label %b.header

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

define void @hoist_inner_loop_switch(i32* %ptr) {
; CHECK-LABEL: define void @hoist_inner_loop_switch(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  %x.a = load i32, i32* %ptr
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    %x.a = load i32, i32* %ptr
; CHECK-NEXT:    br label %b.header

b.header:
  %x.b = load i32, i32* %ptr
  %v1 = call i32 @cond.i32()
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    %x.b = load i32, i32* %ptr
; CHECK-NEXT:    %v1 = call i32 @cond.i32()
; CHECK-NEXT:    switch i32 %v1, label %[[B_HEADER_SPLIT:.*]] [
; CHECK-NEXT:      i32 1, label %[[B_HEADER_SPLIT_US:.*]]
; CHECK-NEXT:      i32 2, label %[[B_HEADER_SPLIT_US]]
; CHECK-NEXT:      i32 3, label %[[B_HEADER_SPLIT_US]]
; CHECK-NEXT:    ]
;
; CHECK:       [[B_HEADER_SPLIT_US]]:
; CHECK-NEXT:    br label %[[C_HEADER_US:.*]]
;
; CHECK:       [[C_HEADER_US]]:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %[[B_LATCH_US:.*]]
;
; CHECK:       [[B_LATCH_US]]:
; CHECK-NEXT:    br label %b.latch
;
; CHECK:       [[B_HEADER_SPLIT]]:
; CHECK-NEXT:    %[[X_A_LCSSA:.*]] = phi i32 [ %x.a, %b.header ]
; CHECK-NEXT:    %[[X_B_LCSSA:.*]] = phi i32 [ %x.b, %b.header ]
; CHECK-NEXT:    br label %c.header

c.header:
  call i32 @c()
  switch i32 %v1, label %c.latch [
    i32 1, label %b.latch
    i32 2, label %b.latch
    i32 3, label %b.latch
  ]
; CHECK:       c.header:
; CHECK-NEXT:    call i32 @c()
; CHECK-NEXT:    br label %c.latch

c.latch:
  ; Use values from other loops to check LCSSA form.
  store i32 %x.a, i32* %ptr
  store i32 %x.b, i32* %ptr
  %v2 = call i1 @cond()
  br i1 %v2, label %c.header, label %exit
; CHECK:       c.latch:
; CHECK-NEXT:    store i32 %[[X_A_LCSSA]], i32* %ptr
; CHECK-NEXT:    store i32 %[[X_B_LCSSA]], i32* %ptr
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %c.header, label %exit

b.latch:
  %v3 = call i1 @cond()
  br i1 %v3, label %b.header, label %a.latch
; CHECK:       b.latch:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %b.header, label %a.latch

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; A devilish pattern. This is a crafty, crafty test case designed to risk
; creating indirect cycles with trivial and non-trivial unswitching. The inner
; loop has a switch with a trivial exit edge that can be unswitched, but the
; rest of the switch cannot be unswitched because its cost is too high.
; However, the unswitching of the trivial edge creates a new switch in the
; outer loop. *This* switch isn't trivial, but has a low cost to unswitch. When
; we unswitch this switch from the outer loop, we will remove it completely and
; create a clone of the inner loop on one side. This clone will then again be
; viable for unswitching the inner-most loop. This lets us check that the
; unswitching doesn't end up cycling infinitely even when the cycle is
; indirect and due to revisiting a loop after cloning.
define void @test31(i32 %arg) {
; CHECK-LABEL: define void @test31(
entry:
  br label %outer.header
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 %arg, label %[[ENTRY_SPLIT:.*]] [
; CHECK-NEXT:      i32 1, label %[[ENTRY_SPLIT_US:.*]]
; CHECK-NEXT:      i32 2, label %[[ENTRY_SPLIT_US]]
; CHECK-NEXT:    ]
;
; CHECK:       [[ENTRY_SPLIT_US]]:
; CHECK-NEXT:    switch i32 %arg, label %[[ENTRY_SPLIT_US_SPLIT:.*]] [
; CHECK-NEXT:      i32 1, label %[[ENTRY_SPLIT_US_SPLIT_US:.*]]
; CHECK-NEXT:    ]

outer.header:
  br label %inner.header

inner.header:
  switch i32 %arg, label %inner.loopexit1 [
    i32 1, label %inner.body1
    i32 2, label %inner.body2
  ]

inner.body1:
  %a = call i32 @a()
  br label %inner.latch
; The (super convoluted) fully unswitched loop around `@a`.
;
; CHECK:       [[ENTRY_SPLIT_US_SPLIT_US]]:
; CHECK-NEXT:    br label %[[OUTER_HEADER_US_US:.*]]
;
; CHECK:       [[OUTER_HEADER_US_US]]:
; CHECK-NEXT:    br label %[[OUTER_HEADER_SPLIT_US_US:.*]]
;
; CHECK:       [[OUTER_LATCH_US_US:.*]]:
; CHECK-NEXT:    %[[OUTER_COND_US_US:.*]] = call i1 @cond()
; CHECK-NEXT:    br i1 %[[OUTER_COND_US_US]], label %[[OUTER_HEADER_US_US]], label %[[EXIT_SPLIT_US_SPLIT_US:.*]]
;
; CHECK:       [[OUTER_HEADER_SPLIT_US_US]]:
; CHECK-NEXT:    br label %[[OUTER_HEADER_SPLIT_SPLIT_US_US_US:.*]]
;
; CHECK:       [[INNER_LOOPEXIT2_US_US:.*]]:
; CHECK-NEXT:    br label %[[OUTER_LATCH_US_US]]
;
; CHECK:       [[OUTER_HEADER_SPLIT_SPLIT_US_US_US]]:
; CHECK-NEXT:    br label %[[INNER_HEADER_US_US_US:.*]]
;
; CHECK:       [[INNER_HEADER_US_US_US]]:
; CHECK-NEXT:    br label %[[INNER_BODY1_US_US_US:.*]]
;
; CHECK:       [[INNER_BODY1_US_US_US]]:
; CHECK-NEXT:    %[[A:.*]] = call i32 @a()
; CHECK-NEXT:    br label %[[INNER_LATCH_US_US_US:.*]]
;
; CHECK:       [[INNER_LATCH_US_US_US]]:
; CHECK-NEXT:    %[[PHI_A:.*]] = phi i32 [ %[[A]], %[[INNER_BODY1_US_US_US]] ]
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 %[[PHI_A]])
; CHECK-NEXT:    %[[INNER_COND_US_US_US:.*]] = call i1 @cond()
; CHECK-NEXT:    br i1 %[[INNER_COND_US_US_US]], label %[[INNER_HEADER_US_US_US]], label %[[INNER_LOOPEXIT2_SPLIT_US_US_US:.*]]
;
; CHECK:       [[INNER_LOOPEXIT2_SPLIT_US_US_US]]:
; CHECK-NEXT:    br label %[[INNER_LOOPEXIT2_US_US]]
;
; CHECK:       [[EXIT_SPLIT_US_SPLIT_US]]:
; CHECK-NEXT:    br label %[[EXIT_SPLIT_US:.*]]


inner.body2:
  %b = call i32 @b()
  br label %inner.latch
; The fully unswitched loop around `@b`.
;
; CHECK:       [[ENTRY_SPLIT_US_SPLIT]]:
; CHECK-NEXT:    br label %[[OUTER_HEADER_US:.*]]
;
; CHECK:       [[OUTER_HEADER_US]]:
; CHECK-NEXT:    br label %[[OUTER_HEADER_SPLIT_US:.*]]
;
; CHECK:       [[INNER_HEADER_US:.*]]:
; CHECK-NEXT:    br label %[[INNER_BODY2_US:.*]]
;
; CHECK:       [[INNER_BODY2_US]]:
; CHECK-NEXT:    %[[B:.*]] = call i32 @b()
; CHECK-NEXT:    br label %[[INNER_LATCH_US:.*]]
;
; CHECK:       [[INNER_LATCH_US]]:
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 0)
; CHECK-NEXT:    call void @sink1(i32 %[[B]])
; CHECK-NEXT:    %[[INNER_COND_US:.*]] = call i1 @cond()
; CHECK-NEXT:    br i1 %[[INNER_COND_US]], label %[[INNER_HEADER_US]], label %[[INNER_LOOPEXIT2_SPLIT_US:.*]]
;
; CHECK:       [[INNER_LOOPEXIT2_SPLIT_US]]:
; CHECK-NEXT:    br label %[[INNER_LOOPEXIT2_US:.*]]
;
; CHECK:       [[OUTER_LATCH_US:.*]]:
; CHECK-NEXT:    %[[OUTER_COND_US:.*]] = call i1 @cond()
; CHECK-NEXT:    br i1 %[[OUTER_COND_US]], label %[[OUTER_HEADER_US]], label %[[EXIT_SPLIT_US_SPLIT:.*]]
;
; CHECK:       [[OUTER_HEADER_SPLIT_US]]:
; CHECK-NEXT:    br label %[[OUTER_HEADER_SPLIT_SPLIT_US:.*]]
;
; CHECK:       [[OUTER_HEADER_SPLIT_SPLIT_US]]:
; CHECK-NEXT:    br label %[[INNER_HEADER_US]]
;
; CHECK:       [[INNER_LOOPEXIT2_US]]:
; CHECK-NEXT:    br label %[[OUTER_LATCH_US]]
;
; CHECK:       [[EXIT_SPLIT_US]]:
; CHECK-NEXT:    br label %exit

inner.latch:
  %phi = phi i32 [ %a, %inner.body1 ], [ %b, %inner.body2 ]
  ; Make 10 junk calls here to ensure we're over the "50" cost threshold of
  ; non-trivial unswitching for this inner switch.
  call void @sink1(i32 0)
  call void @sink1(i32 0)
  call void @sink1(i32 0)
  call void @sink1(i32 0)
  call void @sink1(i32 0)
  call void @sink1(i32 0)
  call void @sink1(i32 0)
  call void @sink1(i32 0)
  call void @sink1(i32 0)
  call void @sink1(i32 0)
  call void @sink1(i32 %phi)
  %inner.cond = call i1 @cond()
  br i1 %inner.cond, label %inner.header, label %inner.loopexit2

inner.loopexit1:
  br label %outer.latch
; The unswitched `loopexit1` path.
;
; CHECK:       [[ENTRY_SPLIT]]:
; CHECK-NEXT:    br label %[[OUTER_HEADER:.*]]
;
; CHECK:       outer.header:
; CHECK-NEXT:    br label %inner.loopexit1
;
; CHECK:       inner.loopexit1:
; CHECK-NEXT:    br label %outer.latch
;
; CHECK:       outer.latch:
; CHECK-NEXT:    %outer.cond = call i1 @cond()
; CHECK-NEXT:    br i1 %outer.cond, label %outer.header, label %[[EXIT_SPLIT:.*]]
;
; CHECK:       [[EXIT_SPLIT]]:
; CHECK-NEXT:    br label %exit

inner.loopexit2:
  br label %outer.latch

outer.latch:
  %outer.cond = call i1 @cond()
  br i1 %outer.cond, label %outer.header, label %exit

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; Non-trivial partial loop unswitching of multiple invariant inputs to an `and`
; chain (select version).
define i32 @test32(i1* %ptr1, i1* %ptr2, i1* %ptr3, i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @test32(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[INV_AND:.*]] = and i1 %cond2, %cond1
; CHECK-NEXT:    br i1 %[[INV_AND]], label %entry.split, label %entry.split.us

loop_begin:
  %v1 = load i1, i1* %ptr1
  %v2 = load i1, i1* %ptr2
  %cond_and1 = select i1 %v1, i1 %cond1, i1 false
  %cond_and2 = select i1 %cond_and1, i1 %cond2, i1 false
  br i1 %cond_and2, label %loop_a, label %loop_b
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[V2_US]] = load i1, i1* %ptr2, align 1
; CHECK-NEXT:    br label %loop_b.us
;
; CHECK:       loop_b.us:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %latch.us
;
; CHECK:       latch.us:
; CHECK-NEXT:    %[[V3_US:.*]] = load i1, i1* %ptr3, align 1
; CHECK-NEXT:    br i1 %[[V3_US]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    br label %loop_exit

; The original loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V1:.*]] = load i1, i1* %ptr1
; CHECK-NEXT:    %[[V2:.*]] = load i1, i1* %ptr2
; CHECK-NEXT:    %[[AND1:.*]] = select i1 %[[V1]], i1 true, i1 false
; CHECK-NEXT:    %[[AND2:.*]] = select i1 %[[AND1]], i1 true, i1 false
; CHECK-NEXT:    br i1 %[[AND2]], label %loop_a, label %loop_b

loop_a:
  call i32 @a()
  br label %latch
; CHECK:       loop_a:
; CHECK-NEXT:    call i32 @a()
; CHECK-NEXT:    br label %latch

loop_b:
  call i32 @b()
  br label %latch
; CHECK:       loop_b:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %latch

latch:
  %v3 = load i1, i1* %ptr3
  br i1 %v3, label %loop_begin, label %loop_exit
; CHECK:       latch:
; CHECK-NEXT:    %[[V3:.*]] = load i1, i1* %ptr3, align 1
; CHECK-NEXT:    br i1 %[[V3]], label %loop_begin, label %loop_exit.split

loop_exit:
  ret i32 0
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    ret
}

; Non-trivial partial loop unswitching of multiple invariant inputs to an `or`
; chain (select version).
define i32 @test33(i1* %ptr1, i1* %ptr2, i1* %ptr3, i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @test33(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[INV_OR:.*]] = or i1 %cond2, %cond1
; CHECK-NEXT:    br i1 %[[INV_OR]], label %entry.split.us, label %entry.split

loop_begin:
  %v1 = load i1, i1* %ptr1
  %v2 = load i1, i1* %ptr2
  %cond_and1 = select i1 %v1, i1 true, i1 %cond1
  %cond_and2 = select i1 %cond_and1, i1 true, i1 %cond2
  br i1 %cond_and2, label %loop_b, label %loop_a
; The 'loop_b' unswitched loop.
;
; CHECK:       entry.split.us:
; CHECK-NEXT:    br label %loop_begin.us
;
; CHECK:       loop_begin.us:
; CHECK-NEXT:    %[[V2_US]] = load i1, i1* %ptr2, align 1
; CHECK-NEXT:    br label %loop_b.us
;
; CHECK:       loop_b.us:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %latch.us
;
; CHECK:       latch.us:
; CHECK-NEXT:    %[[V3_US:.*]] = load i1, i1* %ptr3, align 1
; CHECK-NEXT:    br i1 %[[V3_US]], label %loop_begin.us, label %loop_exit.split.us
;
; CHECK:       loop_exit.split.us:
; CHECK-NEXT:    br label %loop_exit

; The original loop.
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin
;
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V1:.*]] = load i1, i1* %ptr1
; CHECK-NEXT:    %[[V2:.*]] = load i1, i1* %ptr2
; CHECK-NEXT:    %[[AND1:.*]] = select i1 %[[V1]], i1 true, i1 false
; CHECK-NEXT:    %[[AND2:.*]] = select i1 %[[AND1]], i1 true, i1 false
; CHECK-NEXT:    br i1 %[[AND2]], label %loop_b, label %loop_a

loop_a:
  call i32 @a()
  br label %latch
; CHECK:       loop_a:
; CHECK-NEXT:    call i32 @a()
; CHECK-NEXT:    br label %latch

loop_b:
  call i32 @b()
  br label %latch
; CHECK:       loop_b:
; CHECK-NEXT:    call i32 @b()
; CHECK-NEXT:    br label %latch

latch:
  %v3 = load i1, i1* %ptr3
  br i1 %v3, label %loop_begin, label %loop_exit
; CHECK:       latch:
; CHECK-NEXT:    %[[V3:.*]] = load i1, i1* %ptr3, align 1
; CHECK-NEXT:    br i1 %[[V3]], label %loop_begin, label %loop_exit.split

loop_exit:
  ret i32 0
; CHECK:       loop_exit.split:
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    ret
}
