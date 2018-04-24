; RUN: opt -passes='loop(unswitch),verify<loops>' -enable-nontrivial-unswitch -S < %s | FileCheck %s
; RUN: opt -simple-loop-unswitch -enable-nontrivial-unswitch -S < %s | FileCheck %s

declare void @a()
declare void @b()
declare void @c()
declare void @d()

declare void @sink1(i32)
declare void @sink2(i32)

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
  call void @a() convergent
  br label %loop_latch

loop_b:
  call void @b()
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
  call void @a() noduplicate
  br label %loop_latch

loop_b:
  call void @b()
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
  call void @a()
  br label %loop_cont

loop_b:
  call void @b()
  br label %loop_cont

loop_cont:
  invoke void @a()
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
  call void @a()
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
; CHECK-NEXT:    call void @a()
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
  call void @b()
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
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    br label %latch.us2
;
; CHECK:       latch.us2:
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin.us1, label %loop_exit.split.split.us
;
; CHECK:       loop_exit.split.split.us:
; CHECK-NEXT:    br label %loop_exit.split

loop_b_b:
  call void @c()
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
; CHECK-NEXT:    call void @c()
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
; The 'loop_b' unswitched loop.
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
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A]], %loop_begin ]
; CHECK-NEXT:    br label %loop_exit

loop_exit:
  %ab.phi = phi i32 [ %b, %loop_b ], [ %a, %loop_begin ]
  ret i32 %ab.phi
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[AB_PHI:.*]] = phi i32 [ %[[A_LCSSA]], %loop_exit.split ], [ %[[B_LCSSA]], %loop_exit.split.us ]
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
; CHECK-NEXT:    %[[A_PHI:.*]] = phi i32 [ %[[A_LCSSA]], %loop_exit1.split.us ]
; CHECK-NEXT:    call void @sink1(i32 %[[A_PHI]])
; CHECK-NEXT:    ret void

loop_exit2:
  %b.phi = phi i32 [ %b, %loop_b ]
  call void @sink2(i32 %b.phi)
  ret void
; CHECK:       loop_exit2:
; CHECK-NEXT:    %[[B_PHI:.*]] = phi i32 [ %[[B]], %loop_b ]
; CHECK-NEXT:    call void @sink2(i32 %[[B_PHI]])
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
; CHECK-NEXT:    %[[B_PHI:.*]] = phi i32 [ %[[B_LCSSA]], %loop_exit2.split.us ]
; CHECK-NEXT:    call void @sink2(i32 %[[B_PHI]])
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
; CHECK-NEXT:    %[[A_PHI:.*]] = phi i32 [ %[[A_LCSSA]], %loop_exit1.split.us ]
; CHECK-NEXT:    call void @sink1(i32 %[[A_PHI]])
; CHECK-NEXT:    br label %exit

loop_exit2:
  %b.phi = phi i32 [ %b, %loop_b ]
  call void @sink2(i32 %b.phi)
  br label %exit
; CHECK:       loop_exit2:
; CHECK-NEXT:    %[[B_PHI:.*]] = phi i32 [ %[[B]], %loop_b ]
; CHECK-NEXT:    call void @sink2(i32 %[[B_PHI]])
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
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B]], %inner_loop_b ]
; CHECK-NEXT:    %[[V2:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V2]], label %loop_begin, label %loop_exit.loopexit1

loop_exit:
  %ab.phi = phi i32 [ %a, %inner_loop_begin ], [ %b.phi, %loop_latch ]
  ret i32 %ab.phi
; CHECK:       loop_exit.loopexit:
; CHECK-NEXT:    %[[A_PHI:.*]] = phi i32 [ %[[A_LCSSA]], %loop_exit.loopexit.split.us ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit.loopexit1:
; CHECK-NEXT:    %[[B_PHI:.*]] = phi i32 [ %[[B_LCSSA]], %loop_latch ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[AB_PHI:.*]] = phi i32 [ %[[A_PHI]], %loop_exit.loopexit ], [ %[[B_PHI]], %loop_exit.loopexit1 ]
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
; CHECK-NEXT:    %[[B_PHI:.*]] = phi i32 [ %[[B_INNER_LCSSA]], %loop_b_inner_exit ]
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[B_LCSSA:.*]] = phi i32 [ %[[B_PHI]], %latch ]
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
; CHECK-NEXT:    %[[A_INNER_LCSSA:.*]] = phi i32 [ %[[A_INNER_LCSSA_US]], %inner_loop_exit.split.us ]
; CHECK-NEXT:    %[[V:.*]] = load i1, i1* %ptr
; CHECK-NEXT:    br i1 %[[V]], label %loop_begin, label %loop_exit

loop_exit:
  %a.lcssa = phi i32 [ %a.inner_lcssa, %inner_loop_exit ]
  ret i32 %a.lcssa
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A_INNER_LCSSA]], %inner_loop_exit ]
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
; CHECK-NEXT:    ret i32 %[[AB_PHI]]
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
; CHECK-NEXT:    ret i32 %[[AB_PHI]]
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
; CHECK-NEXT:    %[[A_LCSSA_US:.*]] = phi i32 [ %[[A_INNER_LCSSA_US]], %loop_exit.loopexit.split.us ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit.loopexit1:
; CHECK-NEXT:    %[[A_LCSSA:.*]] = phi i32 [ %[[A_INNER_LCSSA]], %inner_loop_exit ]
; CHECK-NEXT:    br label %loop_exit
;
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[A_PHI:.*]] = phi i32 [ %[[A_LCSSA_US]], %loop_exit.loopexit ], [ %[[A_LCSSA]], %loop_exit.loopexit1 ]
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
; CHECK-NEXT:    %[[A_INNER_PHI:.*]] = phi i32 [ %[[A_INNER_LCSSA_US]], %inner_loop_exit.split.us ]
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
; CHECK-NEXT:    %[[A_LCSSA_US:.*]] = phi i32 [ %[[A_INNER_PHI]], %inner_loop_exit ]
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
; CHECK-NEXT:    %[[A_INNER_LCSSA_US:.*]] = phi i32 [ %[[A_INNER_INNER_LCSSA_US]], %inner_loop_exit.loopexit.split.us ]
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit.loopexit1:
; CHECK-NEXT:    %[[A_INNER_LCSSA:.*]] = phi i32 [ %[[A_INNER_INNER_LCSSA]], %inner_inner_loop_exit ]
; CHECK-NEXT:    br label %inner_loop_exit
;
; CHECK:       inner_loop_exit:
; CHECK-NEXT:    %[[A_INNER_PHI:.*]] = phi i32 [ %[[A_INNER_LCSSA_US]], %inner_loop_exit.loopexit ], [ %[[A_INNER_LCSSA]], %inner_loop_exit.loopexit1 ]
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
; CHECK-NEXT:    %[[A_INNER_INNER_PHI:.*]] = phi i32 [ %[[A_INNER_INNER_LCSSA_US]], %inner_inner_loop_exit.split.us ]
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
; CHECK-NEXT:    %[[A_INNER_LCSSA_US:.*]] = phi i32 [ %[[A_INNER_INNER_PHI]], %inner_inner_loop_exit ]
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
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %var_val = load i32, i32* %var
  switch i32 %cond2, label %loop_a [
    i32 0, label %loop_b
    i32 1, label %loop_b
    i32 13, label %loop_c
    i32 2, label %loop_b
    i32 42, label %loop_exit
  ]
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[V:.*]] = load i32, i32* %var
; CHECK-NEXT:    switch i32 %cond2, label %loop_a [
; CHECK-NEXT:      i32 0, label %loop_b
; CHECK-NEXT:      i32 1, label %loop_b
; CHECK-NEXT:      i32 13, label %loop_c
; CHECK-NEXT:      i32 2, label %loop_b
; CHECK-NEXT:      i32 42, label %loop_exit
; CHECK-NEXT:    ]

loop_a:
  call void @a()
  br label %loop_latch
; CHECK:       loop_a:
; CHECK-NEXT:    call void @a()
; CHECK-NEXT:    br label %loop_latch

loop_b:
  call void @b()
  br label %loop_latch
; CHECK:       loop_b:
; CHECK-NEXT:    call void @b()
; CHECK-NEXT:    br label %loop_latch

loop_c:
  call void @c() noreturn nounwind
  br label %loop_latch
; CHECK:       loop_c:
; CHECK-NEXT:    call void @c()
; CHECK-NEXT:    br label %loop_latch

loop_latch:
  br label %loop_begin
; CHECK:       loop_latch:
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  %lcssa = phi i32 [ %var_val, %loop_begin ]
  ret i32 %lcssa
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
