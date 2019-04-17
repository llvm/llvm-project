; RUN: opt < %s -S -loop-simplify | FileCheck %s
; RUN: opt < %s -S -passes=loop-simplify | FileCheck %s

; This function should get a preheader inserted before bb3, that is jumped
; to by bb1 & bb2
define void @test() {
; CHECK-LABEL: define void @test(
entry:
  br i1 true, label %bb1, label %bb2

bb1:
  br label %bb3
; CHECK:      bb1:
; CHECK-NEXT:   br label %[[PH:.*]]

bb2:
  br label %bb3
; CHECK:      bb2:
; CHECK-NEXT:   br label %[[PH]]

bb3:
  br label %bb3
; CHECK:      [[PH]]:
; CHECK-NEXT:   br label %bb3
;
; CHECK:      bb3:
; CHECK-NEXT:   br label %bb3
}

; Test a case where we have multiple exit blocks as successors of a single loop
; block that need to be made dedicated exit blocks. We also have multiple
; exiting edges to one of the exit blocks that all should be rewritten.
define void @test_multiple_exits_from_single_block(i8 %a, i8* %b.ptr) {
; CHECK-LABEL: define void @test_multiple_exits_from_single_block(
entry:
  switch i8 %a, label %loop [
    i8 0, label %exit.a
    i8 1, label %exit.b
  ]
; CHECK:      entry:
; CHECK-NEXT:   switch i8 %a, label %[[PH:.*]] [
; CHECK-NEXT:     i8 0, label %exit.a
; CHECK-NEXT:     i8 1, label %exit.b
; CHECK-NEXT:   ]

loop:
  %b = load volatile i8, i8* %b.ptr
  switch i8 %b, label %loop [
    i8 0, label %exit.a
    i8 1, label %exit.b
    i8 2, label %loop
    i8 3, label %exit.a
    i8 4, label %loop
    i8 5, label %exit.a
    i8 6, label %loop
  ]
; CHECK:      [[PH]]:
; CHECK-NEXT:   br label %loop
;
; CHECK:      loop:
; CHECK-NEXT:   %[[B:.*]] = load volatile i8, i8* %b.ptr
; CHECK-NEXT:   switch i8 %[[B]], label %[[BACKEDGE:.*]] [
; CHECK-NEXT:     i8 0, label %[[LOOPEXIT_A:.*]]
; CHECK-NEXT:     i8 1, label %[[LOOPEXIT_B:.*]]
; CHECK-NEXT:     i8 2, label %[[BACKEDGE]]
; CHECK-NEXT:     i8 3, label %[[LOOPEXIT_A]]
; CHECK-NEXT:     i8 4, label %[[BACKEDGE]]
; CHECK-NEXT:     i8 5, label %[[LOOPEXIT_A]]
; CHECK-NEXT:     i8 6, label %[[BACKEDGE]]
; CHECK-NEXT:   ]
;
; CHECK:      [[BACKEDGE]]:
; CHECK-NEXT:   br label %loop

exit.a:
  ret void
; CHECK:      [[LOOPEXIT_A]]:
; CHECK-NEXT:   br label %exit.a
;
; CHECK:      exit.a:
; CHECK-NEXT:   ret void

exit.b:
  ret void
; CHECK:      [[LOOPEXIT_B]]:
; CHECK-NEXT:   br label %exit.b
;
; CHECK:      exit.b:
; CHECK-NEXT:   ret void
}

; Check that we leave already dedicated exits alone when forming dedicated exit
; blocks.
define void @test_pre_existing_dedicated_exits(i1 %a, i1* %ptr) {
; CHECK-LABEL: define void @test_pre_existing_dedicated_exits(
entry:
  br i1 %a, label %loop.ph, label %non_dedicated_exit
; CHECK:      entry:
; CHECK-NEXT:   br i1 %a, label %loop.ph, label %non_dedicated_exit

loop.ph:
  br label %loop.header
; CHECK:      loop.ph:
; CHECK-NEXT:   br label %loop.header

loop.header:
  %c1 = load volatile i1, i1* %ptr
  br i1 %c1, label %loop.body1, label %dedicated_exit1
; CHECK:      loop.header:
; CHECK-NEXT:   %[[C1:.*]] = load volatile i1, i1* %ptr
; CHECK-NEXT:   br i1 %[[C1]], label %loop.body1, label %dedicated_exit1

loop.body1:
  %c2 = load volatile i1, i1* %ptr
  br i1 %c2, label %loop.body2, label %non_dedicated_exit
; CHECK:      loop.body1:
; CHECK-NEXT:   %[[C2:.*]] = load volatile i1, i1* %ptr
; CHECK-NEXT:   br i1 %[[C2]], label %loop.body2, label %[[LOOPEXIT:.*]]

loop.body2:
  %c3 = load volatile i1, i1* %ptr
  br i1 %c3, label %loop.backedge, label %dedicated_exit2
; CHECK:      loop.body2:
; CHECK-NEXT:   %[[C3:.*]] = load volatile i1, i1* %ptr
; CHECK-NEXT:   br i1 %[[C3]], label %loop.backedge, label %dedicated_exit2

loop.backedge:
  br label %loop.header
; CHECK:      loop.backedge:
; CHECK-NEXT:   br label %loop.header

dedicated_exit1:
  ret void
; Check that there isn't a split loop exit.
; CHECK-NOT:    br label %dedicated_exit1
;
; CHECK:      dedicated_exit1:
; CHECK-NEXT:   ret void

dedicated_exit2:
  ret void
; Check that there isn't a split loop exit.
; CHECK-NOT:    br label %dedicated_exit2
;
; CHECK:      dedicated_exit2:
; CHECK-NEXT:   ret void

non_dedicated_exit:
  ret void
; CHECK:      [[LOOPEXIT]]:
; CHECK-NEXT:   br label %non_dedicated_exit
;
; CHECK:      non_dedicated_exit:
; CHECK-NEXT:   ret void
}

; Check that we form what dedicated exits we can even when some exits are
; reached via indirectbr which precludes forming dedicated exits.
define void @test_form_some_dedicated_exits_despite_indirectbr(i8 %a, i8* %ptr, i8** %addr.ptr) {
; CHECK-LABEL: define void @test_form_some_dedicated_exits_despite_indirectbr(
entry:
  switch i8 %a, label %loop.ph [
    i8 0, label %exit.a
    i8 1, label %exit.b
    i8 2, label %exit.c
  ]
; CHECK:      entry:
; CHECK-NEXT:   switch i8 %a, label %loop.ph [
; CHECK-NEXT:     i8 0, label %exit.a
; CHECK-NEXT:     i8 1, label %exit.b
; CHECK-NEXT:     i8 2, label %exit.c
; CHECK-NEXT:   ]

loop.ph:
  br label %loop.header
; CHECK:      loop.ph:
; CHECK-NEXT:   br label %loop.header

loop.header:
  %addr1 = load volatile i8*, i8** %addr.ptr
  indirectbr i8* %addr1, [label %loop.body1, label %exit.a]
; CHECK:      loop.header:
; CHECK-NEXT:   %[[ADDR1:.*]] = load volatile i8*, i8** %addr.ptr
; CHECK-NEXT:   indirectbr i8* %[[ADDR1]], [label %loop.body1, label %exit.a]

loop.body1:
  %b = load volatile i8, i8* %ptr
  switch i8 %b, label %loop.body2 [
    i8 0, label %exit.a
    i8 1, label %exit.b
    i8 2, label %exit.c
  ]
; CHECK:      loop.body1:
; CHECK-NEXT:   %[[B:.*]] = load volatile i8, i8* %ptr
; CHECK-NEXT:   switch i8 %[[B]], label %loop.body2 [
; CHECK-NEXT:     i8 0, label %exit.a
; CHECK-NEXT:     i8 1, label %[[LOOPEXIT:.*]]
; CHECK-NEXT:     i8 2, label %exit.c
; CHECK-NEXT:   ]

loop.body2:
  %addr2 = load volatile i8*, i8** %addr.ptr
  indirectbr i8* %addr2, [label %loop.backedge, label %exit.c]
; CHECK:      loop.body2:
; CHECK-NEXT:   %[[ADDR2:.*]] = load volatile i8*, i8** %addr.ptr
; CHECK-NEXT:   indirectbr i8* %[[ADDR2]], [label %loop.backedge, label %exit.c]

loop.backedge:
  br label %loop.header
; CHECK:      loop.backedge:
; CHECK-NEXT:   br label %loop.header

exit.a:
  ret void
; Check that there isn't a split loop exit.
; CHECK-NOT:    br label %exit.a
;
; CHECK:      exit.a:
; CHECK-NEXT:   ret void

exit.b:
  ret void
; CHECK:      [[LOOPEXIT]]:
; CHECK-NEXT:   br label %exit.b
;
; CHECK:      exit.b:
; CHECK-NEXT:   ret void

exit.c:
  ret void
; Check that there isn't a split loop exit.
; CHECK-NOT:    br label %exit.c
;
; CHECK:      exit.c:
; CHECK-NEXT:   ret void
}
