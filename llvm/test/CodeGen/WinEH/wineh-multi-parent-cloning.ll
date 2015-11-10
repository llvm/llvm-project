; RUN: opt -mtriple=x86_x64-pc-windows-msvc -S -winehprepare  < %s | FileCheck %s

declare i32 @__CxxFrameHandler3(...)

declare void @f()
declare i32 @g()
declare void @h(i32)
declare i1 @b()

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %invoke.cont unwind label %left
invoke.cont:
  invoke void @f()
    to label %exit unwind label %right
left:
  cleanuppad []
  br label %shared
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind to caller
shared:
  %x = call i32 @g()
  invoke void @f()
    to label %shared.cont unwind label %inner
shared.cont:
  unreachable
inner:
  %i = cleanuppad []
  call void @h(i32 %x)
  cleanupret %i unwind label %right.end
exit:
  ret void
}
; %inner is a cleanup which appears both as a child of
; %left and as a child of %right.  Since statically we
; need each funclet to have a single parent, we need to
; clone the entire %inner funclet so we can have one
; copy under each parent.  The cleanupret in %inner
; unwinds to the catchendpad for %right, so the copy
; of %inner under %right should include it; the copy
; of %inner under %left should instead have an
; `unreachable` inserted there, but the copy under
; %left still needs to be created because it's possible
; the dynamic path enters %left, then enters %inner,
; then calls @h, and that the call to @h doesn't return.
; CHECK-LABEL: define void @test1(
; CHECK:     left:
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     right:
; CHECK:           to label %right.catch unwind label %right.end
; CHECK:     right.catch:
; CHECK:       %x = call i32 @g()
; CHECK:       store i32 %x, i32* %x.wineh.spillslot
; CHECK:           to label %shared.cont unwind label %[[INNER_RIGHT:.+]]
; CHECK:     right.end:
; CHECK:       catchendpad unwind to caller
; CHECK:     shared.cont:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       [[I_R:\%.+]] = cleanuppad []
; CHECK:       [[X_RELOAD_R:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_R]])
; CHECK:       cleanupret [[I_R]] unwind label %right.end
; CHECK:     [[INNER_LEFT]]:
; CHECK:       [[I_L:\%.+]] = cleanuppad []
; CHECK:       [[X_RELOAD_L:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_L]])
; CHECK:       unreachable


define void @test2() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %invoke.cont unwind label %left
invoke.cont:
  invoke void @f()
    to label %exit unwind label %right
left:
  cleanuppad []
  br label %shared
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind to caller
shared:
  %x = call i32 @g()
  invoke void @f()
    to label %shared.cont unwind label %inner
shared.cont:
  unreachable
inner:
  catchpad []
    to label %inner.catch unwind label %inner.end
inner.catch:
  call void @h(i32 %x)
  unreachable
inner.end:
  catchendpad unwind label %right.end
exit:
  ret void
}
; In this case left and right are both parents of inner.  This differs from
; @test1 in that inner is a catchpad rather than a cleanuppad, which makes
; inner.end a block that gets cloned so that left and right each contain a
; copy (catchendpad blocks are considered to be part of the parent funclet
; of the associated catchpad). The catchendpad in %inner.end unwinds to
; %right.end (which belongs to the entry funclet).
; CHECK-LABEL: define void @test2(
; CHECK:     left:
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     right:
; CHECK:           to label %right.catch unwind label %[[RIGHT_END:.+]]
; CHECK:     right.catch:
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[RIGHT_END]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       [[X_RELOAD_R:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_R]])
; CHECK:       unreachable
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       [[X_RELOAD_L:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_L]])
; CHECK:       unreachable
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind label %[[RIGHT_END]]

define void @test3() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %left
left:
  %l = cleanuppad []
  br label %shared
left.end:
  cleanupendpad %l unwind label %right
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind to caller
shared:
  %x = call i32 @g()
  invoke void @f()
    to label %shared.cont unwind label %inner
shared.cont:
  unreachable
inner:
  catchpad []
    to label %inner.catch unwind label %inner.end
inner.catch:
  call void @h(i32 %x)
  unreachable
inner.end:
  catchendpad unwind label %left.end
exit:
  ret void
}
; In this case, %left and %right are siblings with %entry as the parent of both,
; while %left and %right are both parents of %inner.  The catchendpad in
; %inner.end unwinds to %left.end.  When %inner is cloned a copy of %inner.end
; will be made for both %left and %right, but because %left.end is a cleanup pad
; and %right is a catch pad the unwind edge from the copy of %inner.end for
; %right must be removed.
; CHECK-LABEL: define void @test3(
; CHECK:     left:
; CHECK:       %l = cleanuppad []
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     [[LEFT_END:left.end.*]]:
; CHECK:       cleanupendpad %l unwind label %right
; CHECK:     right:
; CHECK:           to label %right.catch unwind label %[[RIGHT_END:.+]]
; CHECK:     right.catch:
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[RIGHT_END]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       [[X_RELOAD_R:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_R]])
; CHECK:       unreachable
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       [[X_RELOAD_R:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_R]])
; CHECK:       unreachable
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind label %[[LEFT_END]]
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind to caller


define void @test4() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %left
left:
  catchpad []
    to label %left.catch unwind label %left.end
left.catch:
  br label %shared
left.end:
  catchendpad unwind label %right
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind to caller
shared:
  %x = call i32 @g()
  invoke void @f()
    to label %shared.cont unwind label %inner
shared.cont:
  unreachable
inner:
  catchpad []
    to label %inner.catch unwind label %inner.end
inner.catch:
  call void @h(i32 %x)
  unreachable
inner.end:
  catchendpad unwind label %left.end
exit:
  ret void
}
; This is a variation of @test3 in which both %left and %right are catch pads.
; In this case, %left and %right are siblings with %entry as the parent of both,
; while %left and %right are both parents of %inner.  The catchendpad in
; %inner.end unwinds to %left.end.  When %inner is cloned a copy of %inner.end
; will be made for both %left and %right, but because the catchpad in %right
; does not unwind to %left.end the unwind edge from the copy of %inner.end for
; %right must be removed.
; CHECK-LABEL: define void @test4(
; CHECK:     left:
; CHECK:       catchpad []
; CHECK:           to label %left.catch unwind label %[[LEFT_END:.+]]
; CHECK:     left.catch:
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     [[LEFT_END]]:
; CHECK:       catchendpad unwind label %right
; CHECK:     right:
; CHECK:           to label %right.catch unwind label %[[RIGHT_END:.+]]
; CHECK:     right.catch:
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[RIGHT_END]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       [[X_RELOAD_R:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_R]])
; CHECK:       unreachable
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       [[X_RELOAD_L:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_L]])
; CHECK:       unreachable
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind label %[[LEFT_END]]


define void @test5() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %left
left:
  catchpad []
    to label %left.catch unwind label %left.end
left.catch:
  br label %shared
left.end:
  catchendpad unwind label %right
right:
  %r = cleanuppad []
  br label %shared
shared:
  %x = call i32 @g()
  invoke void @f()
    to label %shared.cont unwind label %inner
shared.cont:
  unreachable
inner:
  catchpad []
    to label %inner.catch unwind label %inner.end
inner.catch:
  call void @h(i32 %x)
  unreachable
inner.end:
  catchendpad unwind label %left.end
exit:
  ret void
}
; Like @test3, %left and %right are siblings with %entry as the parent of both,
; while %left and %right are both parents of %inner.  This case makes %left a
; catch and %right a cleanup so that %inner unwinds to %left.end, which is a
; block in %entry.  The %inner funclet is cloned for %left and %right, but the
; copy of %inner.end for %right must have its unwind edge removed because the
; catchendpad at %left.end is not compatible with %right.
; CHECK-LABEL: define void @test5(
; CHECK:     left:
; CHECK:       catchpad []
; CHECK:           to label %left.catch unwind label %[[LEFT_END:.+]]
; CHECK:     left.catch:
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     [[LEFT_END]]:
; CHECK:       catchendpad unwind label %right
; CHECK:     right:
; CHECK:       %r = cleanuppad []
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       [[X_RELOAD_R:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_R]])
; CHECK:       unreachable
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       [[X_RELOAD_L:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_L]])
; CHECK:       unreachable
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind label %[[LEFT_END]]

define void @test6() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %left
left:
  catchpad []
    to label %left.catch unwind label %left.end
left.catch:
  br label %shared
left.end:
  catchendpad unwind label %middle
middle:
  %m = catchpad []
    to label %middle.catch unwind label %middle.end
middle.catch:
  catchret %m to label %exit
middle.end:
  catchendpad unwind label %right
right:
  %r = cleanuppad []
  br label %shared
shared:
  %x = call i32 @g()
  invoke void @f()
    to label %shared.cont unwind label %inner
shared.cont:
  unreachable
inner:
  catchpad []
    to label %inner.catch unwind label %inner.end
inner.catch:
  call void @h(i32 %x)
  unreachable
inner.end:
  catchendpad unwind label %left.end
exit:
  ret void
}
; This is like @test5 but it inserts another sibling between %left and %right.
; In this case %left, %middle and %right are all siblings, while %left and
; %right are both parents of %inner.  This checks the proper handling of the
; catchendpad in %inner.end (which will be cloned so that %left and %right both
; have copies) unwinding to a catchendpad that unwinds to a sibling.
; CHECK-LABEL: define void @test6(
; CHECK:     left:
; CHECK:       catchpad []
; CHECK:           to label %left.catch unwind label %[[LEFT_END:.+]]
; CHECK:     left.catch:
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     [[LEFT_END]]:
; CHECK:       catchendpad unwind label %middle
; CHECK:     middle:
; CHECK:       catchpad []
; CHECK:         to label %middle.catch unwind label %middle.end
; CHECK:     middle.catch:
; CHECK:       catchret %m to label %exit
; CHECK:     middle.end:
; CHECK:       catchendpad unwind label %right
; CHECK:     right:
; CHECK:       %r = cleanuppad []
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       [[X_RELOAD_R:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_R]])
; CHECK:       unreachable
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       [[X_RELOAD_L:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_L]])
; CHECK:       unreachable
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind label %[[LEFT_END]]


define void @test7() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %left
left:
  catchpad []
    to label %left.catch unwind label %left.end
left.catch:
  br label %shared
left.end:
  catchendpad unwind label %right
right:
  %r = cleanuppad []
  br label %shared
shared:
  %x = call i32 @g()
  invoke void @f()
    to label %shared.cont unwind label %inner
shared.cont:
  unreachable
inner:
  catchpad []
    to label %inner.catch unwind label %inner.end
inner.catch:
  call void @h(i32 %x)
  unreachable
inner.end:
  catchendpad unwind label %inner.sibling
inner.sibling:
  %is = cleanuppad []
  call void @h(i32 0)
  cleanupret %is unwind label %left.end
exit:
  ret void
}
; This is like @test5 but instead of unwinding to %left.end, the catchendpad
; in %inner.end unwinds to a sibling cleanup pad. Both %inner (along with its
; associated blocks) and %inner.sibling must be cloned for %left and %right.
; The clones of %inner will be identical, but the copy of %inner.sibling for
; %right must end with an unreachable instruction, because it cannot unwind to
; %left.end.
; CHECK-LABEL: define void @test7(
; CHECK:     left:
; CHECK:       catchpad []
; CHECK:           to label %left.catch unwind label %[[LEFT_END:.+]]
; CHECK:     left.catch:
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     [[LEFT_END]]:
; CHECK:       catchendpad unwind label %[[RIGHT:.+]]
; CHECK:     [[RIGHT]]:
; CHECK:       [[R:\%.+]] = cleanuppad []
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       [[X_RELOAD_R:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_R]])
; CHECK:       unreachable
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       [[X_RELOAD_L:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_L]])
; CHECK:       unreachable
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind label %[[INNER_SIBLING_RIGHT:.+]]
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind label %[[INNER_SIBLING_LEFT:.+]]
; CHECK:     [[INNER_SIBLING_RIGHT]]
; CHECK:       [[IS_R:\%.+]] = cleanuppad []
; CHECK:       call void @h(i32 0)
; CHECK:       unreachable
; CHECK:     [[INNER_SIBLING_LEFT]]
; CHECK:       [[IS_L:\%.+]] = cleanuppad []
; CHECK:       call void @h(i32 0)
; CHECK:       cleanupret [[IS_L]] unwind label %[[LEFT_END]]


define void @test8() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %invoke.cont unwind label %left
invoke.cont:
  invoke void @f()
    to label %unreachable unwind label %right
left:
  cleanuppad []
  invoke void @f() to label %unreachable unwind label %inner
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  invoke void @f() to label %unreachable unwind label %inner
right.end:
  catchendpad unwind to caller
inner:
  %i = cleanuppad []
  %x = call i32 @g()
  call void @h(i32 %x)
  cleanupret %i unwind label %right.end
unreachable:
  unreachable
}
; Another case of a two-parent child (like @test1), this time
; with the join at the entry itself instead of following a
; non-pad join.
; CHECK-LABEL: define void @test8(
; CHECK:     invoke.cont:
; CHECK:           to label %[[UNREACHABLE_ENTRY:.+]] unwind label %right
; CHECK:     left:
; CHECK:           to label %[[UNREACHABLE_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     right:
; CHECK:           to label %right.catch unwind label %right.end
; CHECK:     right.catch:
; CHECK:           to label %unreachable unwind label %[[INNER_RIGHT:.+]]
; CHECK:     right.end:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       [[I_R:\%.+]] = cleanuppad []
; CHECK:       [[X_R:\%.+]] = call i32 @g()
; CHECK:       call void @h(i32 [[X_R]])
; CHECK:       cleanupret [[I_R]] unwind label %right.end
; CHECK:     [[INNER_LEFT]]:
; CHECK:       [[I_L:\%.+]] = cleanuppad []
; CHECK:       [[X_L:\%.+]] = call i32 @g()
; CHECK:       call void @h(i32 [[X_L]])
; CHECK:       unreachable
; CHECK:     unreachable:
; CHECK:       unreachable
; CHECK:     [[UNREACHABLE_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[UNREACHABLE_ENTRY]]:
; CHECK:       unreachable


define void @test9() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %invoke.cont unwind label %left
invoke.cont:
  invoke void @f()
    to label %unreachable unwind label %right
left:
  cleanuppad []
  br label %shared
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind to caller
shared:
  invoke void @f()
    to label %unreachable unwind label %inner
inner:
  cleanuppad []
  invoke void @f()
    to label %unreachable unwind label %inner.child
inner.child:
  cleanuppad []
  %x = call i32 @g()
  call void @h(i32 %x)
  unreachable
unreachable:
  unreachable
}
; %inner is a two-parent child which itself has a child; need
; to make two copies of both the %inner and %inner.child.
; CHECK-LABEL: define void @test9(
; CHECK:     invoke.cont:
; CHECK:               to label %[[UNREACHABLE_ENTRY:.+]] unwind label %right
; CHECK:     left:
; CHECK:               to label %[[UNREACHABLE_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     right:
; CHECK:               to label %right.catch unwind label %right.end
; CHECK:     right.catch:
; CHECK:               to label %[[UNREACHABLE_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     right.end:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[INNER_RIGHT]]:
; CHECK:               to label %[[UNREACHABLE_INNER_RIGHT:.+]] unwind label %[[INNER_CHILD_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:               to label %[[UNREACHABLE_INNER_LEFT:.+]] unwind label %[[INNER_CHILD_LEFT:.+]]
; CHECK:     [[INNER_CHILD_RIGHT]]:
; CHECK:       [[TMP:\%.+]] = cleanuppad []
; CHECK:       [[X:\%.+]] = call i32 @g()
; CHECK:       call void @h(i32 [[X]])
; CHECK:       unreachable
; CHECK:     [[INNER_CHILD_LEFT]]:
; CHECK:       [[TMP:\%.+]] = cleanuppad []
; CHECK:       [[X:\%.+]] = call i32 @g()
; CHECK:       call void @h(i32 [[X]])
; CHECK:       unreachable
; CHECK:     [[UNREACHABLE_INNER_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[UNREACHABLE_INNER_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[UNREACHABLE_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[UNREACHABLE_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[UNREACHABLE_ENTRY]]:
; CHECK:       unreachable


define void @test10() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %invoke.cont unwind label %left
invoke.cont:
  invoke void @f()
    to label %unreachable unwind label %right
left:
  cleanuppad []
  call void @h(i32 1)
  invoke void @f()
    to label %unreachable unwind label %right
right:
  cleanuppad []
  call void @h(i32 2)
  invoke void @f()
    to label %unreachable unwind label %left
unreachable:
  unreachable
}
; This is an irreducible loop with two funclets that enter each other;
; need to make two copies of each funclet (one a child of root, the
; other a child of the opposite funclet), but also make sure not to
; clone self-descendants (if we tried to do that we'd need to make an
; infinite number of them).  Presumably if optimizations ever generated
; such a thing it would mean that one of the two cleanups was originally
; the parent of the other, but that we'd somehow lost track in the CFG
; of which was which along the way; generating each possibility lets
; whichever case was correct execute correctly.
; CHECK-LABEL: define void @test10(
; CHECK:     entry:
; CHECK:               to label %invoke.cont unwind label %[[LEFT:.+]]
; CHECK:     invoke.cont:
; CHECK:               to label %[[UNREACHABLE_ENTRY:.+]] unwind label %[[RIGHT:.+]]
; CHECK:     [[LEFT_FROM_RIGHT:.+]]:
; CHECK:       call void @h(i32 1)
; CHECK:       call void @f()
; CHECK:       unreachable
; CHECK:     [[LEFT]]:
; CHECK:       call void @h(i32 1)
; CHECK:       invoke void @f()
; CHECK:               to label %[[UNREACHABLE_LEFT:.+]] unwind label %[[RIGHT_FROM_LEFT:.+]]
; CHECK:     [[RIGHT]]:
; CHECK:       call void @h(i32 2)
; CHECK:       invoke void @f()
; CHECK:               to label %[[UNREACHABLE_RIGHT:.+]] unwind label %[[LEFT_FROM_RIGHT]]
; CHECK:     [[RIGHT_FROM_LEFT]]:
; CHECK:       call void @h(i32 2)
; CHECK:       call void @f()
; CHECK:       unreachable
; CHECK:     [[UNREACHABLE_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[UNREACHABLE_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[UNREACHABLE_ENTRY]]:
; CHECK:       unreachable


define void @test11() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %left
left:
  catchpad []
    to label %left.catch unwind label %left.sibling
left.catch:
  br label %shared
left.sibling:
  %ls = catchpad []
    to label %left.sibling.catch unwind label %left.end
left.sibling.catch:
  catchret %ls to label %exit
left.end:
  catchendpad unwind label %right
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind to caller
shared:
  %x = call i32 @g()
  invoke void @f()
    to label %shared.cont unwind label %inner
shared.cont:
  unreachable
inner:
  catchpad []
    to label %inner.catch unwind label %inner.end
inner.catch:
  call void @h(i32 %x)
  unreachable
inner.end:
  catchendpad unwind label %left.end
exit:
  ret void
}
; This is a variation of @test4 in which the shared child funclet unwinds to a
; catchend pad that is the unwind destination of %left.sibling rather than %left
; but is still a valid destination for %inner as reach from %left.
; When %inner is cloned a copy of %inner.end will be made for both %left and
; %right, but because the catchpad in %right does not unwind to %left.end the
; unwind edge from the copy of %inner.end for %right must be removed.
; CHECK-LABEL: define void @test11(
; CHECK:     left:
; CHECK:       catchpad []
; CHECK:           to label %left.catch unwind label %left.sibling
; CHECK:     left.catch:
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     left.sibling:
; CHECK:       catchpad []
; CHECK:           to label %left.sibling.catch unwind label %[[LEFT_END:.+]]
; CHECK:     [[LEFT_END]]:
; CHECK:       catchendpad unwind label %right
; CHECK:     right:
; CHECK:           to label %right.catch unwind label %[[RIGHT_END:.+]]
; CHECK:     right.catch:
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[RIGHT_END]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       [[X_RELOAD_R:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_R]])
; CHECK:       unreachable
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       [[X_RELOAD_L:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_L]])
; CHECK:       unreachable
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind label %[[LEFT_END]]


define void @test12() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %left
left:
  catchpad []
    to label %left.catch unwind label %right
left.catch:
  br label %shared
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind to caller
shared:
  %x = call i32 @g()
  invoke void @f()
    to label %shared.cont unwind label %inner
shared.cont:
  unreachable
inner:
  catchpad []
    to label %inner.catch unwind label %inner.end
inner.catch:
  call void @h(i32 %x)
  unreachable
inner.end:
  catchendpad unwind label %right.end
exit:
  ret void
}
; In this case %left and %right are both parents of %inner, so %inner must be
; cloned but the catchendpad unwind target in %inner.end is valid for both
; parents, so the unwind edge should not be removed in either case.
; CHECK-LABEL: define void @test12(
; CHECK:     left:
; CHECK:       catchpad []
; CHECK:           to label %left.catch unwind label %right
; CHECK:     left.catch:
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     right:
; CHECK:           to label %right.catch unwind label %[[RIGHT_END:.+]]
; CHECK:     right.catch:
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[RIGHT_END]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       [[X_RELOAD_R:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_R]])
; CHECK:       unreachable
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       [[X_RELOAD_L:\%.+]] = load i32, i32* %x.wineh.spillslot
; CHECK:       call void @h(i32 [[X_RELOAD_L]])
; CHECK:       unreachable
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind label %[[RIGHT_END]]
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind label %[[RIGHT_END]]

define void @test13() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %invoke.cont unwind label %left
invoke.cont:
  invoke void @f()
    to label %exit unwind label %right
left:
  %l = catchpad []
    to label %left.cont unwind label %left.end
left.cont:
  invoke void @f()
    to label %left.ret unwind label %inner
left.ret:
  catchret %l to label %invoke.cont
left.end:
  catchendpad unwind to caller
right:
  %r = catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  invoke void @f()
    to label %right.ret unwind label %inner
right.ret:
  catchret %r to label %exit
right.end:
  catchendpad unwind to caller
shared:
  call void @h(i32 0)
  unreachable
inner:
  %i = catchpad []
    to label %inner.catch unwind label %inner.end
inner.catch:
  call void @h(i32 1)
  catchret %i to label %shared
inner.end:
  catchendpad unwind label %left.end
exit:
  ret void
}
; This case tests the scenario where a funclet with multiple parents uses a
; catchret to return to a block that may exist in either parent funclets.
; Both %left and %right are parents of %inner.  During common block cloning
; a clone of %shared will be made so that both %left and %right have a copy,
; but the copy of %shared for one of the parent funclets will be unreachable
; until the %inner funclet is cloned.  When the %inner.catch block is cloned
; during the %inner funclet cloning, the catchret instruction should be updated
; so that the catchret in the copy %inner.catch for %left returns to the copy of
; %shared in %left and the catchret in the copy of %inner.catch for %right
; returns to the copy of %shared for %right.
; CHECK-LABEL: define void @test13(
; CHECK:     left:
; CHECK:       %l = catchpad []
; CHECK:           to label %left.cont unwind label %left.end
; CHECK:     left.cont:
; CHECK:       invoke void @f()
; CHECK:           to label %left.ret unwind label %[[INNER_LEFT:.+]]
; CHECK:     left.ret:
; CHECK:       catchret %l to label %invoke.cont
; CHECK:     left.end:
; CHECK:       catchendpad unwind to caller
; CHECK:     right:
; CHECK:       %r = catchpad []
; CHECK:           to label %right.catch unwind label %right.end
; CHECK:     right.catch:
; CHECK:       invoke void @f()
; CHECK:           to label %right.ret unwind label %[[INNER_RIGHT:.+]]
; CHECK:     right.ret:
; CHECK:       catchret %r to label %exit
; CHECK:     right.end:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[SHARED_RIGHT:.+]]:
; CHECK:       call void @h(i32 0)
; CHECK:       unreachable
; CHECK:     [[SHARED_LEFT:.+]]:
; CHECK:       call void @h(i32 0)
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       %[[I_RIGHT:.+]] = catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       %[[I_LEFT:.+]] = catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       call void @h(i32 1)
; CHECK:       catchret %[[I_RIGHT]] to label %[[SHARED_RIGHT]]
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       call void @h(i32 1)
; CHECK:       catchret %[[I_LEFT]] to label %[[SHARED_LEFT]]
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind label %[[LEFT_END]]
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind to caller


define void @test14() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %left
left:
  %l = catchpad []
    to label %shared unwind label %left.end
left.cont:
  invoke void @f()
    to label %left.ret unwind label %right
left.ret: 
  catchret %l to label %exit
left.end:
  catchendpad unwind to caller
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind label %left.end
shared:
  invoke void @f()
    to label %shared.cont unwind label %inner
shared.cont:
  unreachable
inner:
  %i = catchpad []
    to label %inner.catch unwind label %inner.end
inner.catch:
  call void @h(i32 0)
  catchret %i to label %left.cont
inner.end:
  catchendpad unwind label %left.end
exit:
  ret void
}
; This case tests another scenario where a funclet with multiple parents uses a
; catchret to return to a block in one of the parent funclets.  Here %right and
; %left are both parents of %inner and %left is a parent of %right.  The
; catchret in %inner.catch will cause %left.cont and %left.ret to be cloned for
; both %left and %right, but the catchret in %left.ret is invalid for %right
; but the catchret instruction in the copy of %left.ret for %right will be
; removed as an implausible terminator.
; CHECK-LABEL: define void @test14(
; CHECK:     left:
; CHECK:       %l = catchpad []
; CHECK:           to label %[[SHARED_LEFT:.+]] unwind label %[[LEFT_END:.+]]
; CHECK:     [[LEFT_CONT:left.cont.*]]:
; CHECK:       invoke void @f()
; CHECK:           to label %[[LEFT_RET:.+]] unwind label %[[RIGHT:.+]]
; CHECK:     [[LEFT_RET]]:
; CHECK:       catchret %l to label %exit
; CHECK:     [[LEFT_END]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[RIGHT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[RIGHT_CATCH:.+]] unwind label %[[RIGHT_END:.+]]
; CHECK:     [[RIGHT_CATCH]]:
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[RIGHT_END]]:
; CHECK:       catchendpad unwind label %[[LEFT_END]]
; CHECK:     [[SHARED_LEFT]]:
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_LEFT]]:
; CHECK:       [[I_LEFT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       [[I_RIGHT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       call void @h(i32 0)
; CHECK:       catchret [[I_LEFT]] to label %[[LEFT_CONT]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       call void @h(i32 0)
; CHECK:       unreachable
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind label %[[LEFT_END]]
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind to caller

define void @test15() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %left
left:
  %l = catchpad []
    to label %left.catch unwind label %left.end
left.catch:
  invoke void @f()
    to label %shared unwind label %right
left.ret:
  catchret %l to label %exit
left.end:
  catchendpad unwind to caller
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind label %left.end
shared:
  invoke void @f()
    to label %shared.cont unwind label %inner
shared.cont:
  unreachable
inner:
  %i = catchpad []
    to label %inner.catch unwind label %inner.end
inner.catch:
  call void @h(i32 0)
  catchret %i to label %left.ret
inner.end:
  catchendpad unwind label %left.end
exit:
  ret void
}
; This case is a variation of test14 but instead of returning to an invoke the
; catchret in %inner.catch returns to a catchret instruction.
; CHECK-LABEL: define void @test15(
; CHECK:     left:
; CHECK:       %l = catchpad []
; CHECK:           to label %left.catch unwind label %[[LEFT_END:.+]]
; CHECK:     left.catch:
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_LEFT:.+]] unwind label %[[RIGHT:.+]]
; CHECK:     [[LEFT_RET_RIGHT:.+]]:
; CHECK:       unreachable
; CHECK:     [[LEFT_RET_LEFT:.+]]:
; CHECK:       catchret %l to label %exit
; CHECK:     [[LEFT_END]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[RIGHT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[RIGHT_CATCH:.+]] unwind label %[[RIGHT_END:.+]]
; CHECK:     [[RIGHT_CATCH]]:
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[RIGHT_END]]:
; CHECK:       catchendpad unwind label %[[LEFT_END]]
; CHECK:     [[SHARED_LEFT]]:
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_LEFT]]:
; CHECK:       [[I_LEFT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       [[I_RIGHT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       call void @h(i32 0)
; CHECK:       catchret [[I_LEFT]] to label %[[LEFT_RET_LEFT]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       call void @h(i32 0)
; CHECK:       catchret [[I_RIGHT]] to label %[[LEFT_RET_RIGHT]]
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind label %[[LEFT_END]]


define void @test16() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %left
left:
  %l = cleanuppad []
  br label %shared
left.cont:
  cleanupret %l unwind label %right
left.end:
  cleanupendpad %l unwind label %right
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind to caller
shared:
  invoke void @f()
    to label %shared.cont unwind label %inner
shared.cont:
  unreachable
inner:
  %i = catchpad []
    to label %inner.catch unwind label %inner.end
inner.catch:
  call void @h(i32 0)
  catchret %i to label %left.cont
inner.end:
  catchendpad unwind label %left.end
exit:
  ret void
}
; This case is another variation of test14 but here the catchret in %inner.catch
; returns to a cleanupret instruction.
; CHECK-LABEL: define void @test16(
; CHECK:     left:
; CHECK:       %l = cleanuppad []
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     [[LEFT_CONT_RIGHT:.+]]:
; CHECK:       unreachable
; CHECK:     [[LEFT_CONT_LEFT:.+]]:
; CHECK:       cleanupret %l unwind label %[[RIGHT:.+]]
; CHECK:     [[LEFT_END_LEFT:.+]]:
; CHECK:       cleanupendpad %l unwind label %[[RIGHT]]
; CHECK:     [[RIGHT]]:
; CHECK:       catchpad []
; CHECK:           to label %[[RIGHT_CATCH:.+]] unwind label %[[RIGHT_END:.+]]
; CHECK:     [[RIGHT_CATCH]]:
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[RIGHT_END]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       [[I_RIGHT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       [[I_LEFT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       call void @h(i32 0)
; CHECK:       catchret [[I_RIGHT]] to label %[[LEFT_CONT_RIGHT]]
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       call void @h(i32 0)
; CHECK:       catchret [[I_LEFT]] to label %[[LEFT_CONT_LEFT]]
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind label %[[LEFT_END_LEFT]]
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind to caller


define void @test17() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %invoke.cont unwind label %left
invoke.cont:
  invoke void @f()
    to label %exit unwind label %right
left:
  %l = cleanuppad []
  br label %shared
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind to caller
shared:
  invoke void @f()
    to label %unreachable unwind label %inner
unreachable:
  unreachable
inner:
  %i = catchpad []
    to label %inner.catch unwind label %inner.sibling
inner.catch:
  call void @h(i32 0)
  unreachable
inner.sibling:
  %is = catchpad []
    to label %inner.sibling.catch unwind label %inner.end
inner.sibling.catch:
  invoke void @f()
    to label %unreachable unwind label %inner.end
inner.end:
  catchendpad unwind label %right.end
exit:
  ret void
}
; This case tests the scenario where two catchpads with the same catchendpad
; have multiple parents.  Both %left and %right are parents of %inner and
; %inner.sibling so both of the inner funclets must be cloned.  Because
; the catchendpad in %inner.end unwinds to the catchendpad for %right, the
; unwind edge should be removed for the copy of %inner.end that is reached
; from %left.  In addition, the %inner.siblin.catch block contains an invoke
; that unwinds to the shared inner catchendpad.  The unwind destination for
; this invoke should be updated to unwind to the correct cloned %inner.end
; for each path to the funclet.
; CHECK-LABEL: define void @test17(
; CHECK:     left:
; CHECK:       %l = cleanuppad []
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     right:
; CHECK:       catchpad []
; CHECK:           to label %[[RIGHT_CATCH:.+]] unwind label %[[RIGHT_END:.+]]
; CHECK:     [[RIGHT_CATCH]]:
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[RIGHT_END]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       [[I_RIGHT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_SIBLING_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       [[I_LEFT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_SIBLING_LEFT:.+]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       call void @h(i32 0)
; CHECK:       unreachable
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       call void @h(i32 0)
; CHECK:       unreachable
; CHECK:     [[INNER_SIBLING_RIGHT]]:
; CHECK:       [[IS_RIGHT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_SIBLING_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_SIBLING_LEFT]]:
; CHECK:       [[IS_LEFT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_SIBLING_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_SIBLING_CATCH_RIGHT]]:
; CHECK:       invoke void @f()
; CHECK:         to label %[[UNREACHABLE_RIGHT:.+]] unwind label %[[INNER_END_RIGHT]]
; CHECK:     [[INNER_SIBLING_CATCH_LEFT]]:
; CHECK:       invoke void @f()
; CHECK:         to label %[[UNREACHABLE_LEFT:.+]] unwind label %[[INNER_END_LEFT]]
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind label %[[RIGHT_END]]


define void @test18() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %invoke.cont unwind label %left
invoke.cont:
  invoke void @f()
    to label %exit unwind label %right
left:
  %l = cleanuppad []
  br label %shared
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind to caller
shared:
  invoke void @f()
    to label %unreachable unwind label %inner
unreachable:
  unreachable
inner:
  %i = catchpad []
    to label %inner.catch unwind label %inner.sibling
inner.catch:
  invoke void @f()
    to label %unreachable unwind label %inner.end
inner.sibling:
  %is = catchpad []
    to label %inner.sibling.catch unwind label %inner.end
inner.sibling.catch:
  call void @h(i32 0)
  unreachable
inner.end:
  catchendpad unwind label %right.end
exit:
  ret void
}
; This is like test17 except that the inner invoke is moved from the
; %inner.sibling funclet to %inner so that it is unwinding to a
; catchendpad block that has not yet been cloned.  The unwind destination
; of the invoke should still be updated to reach the correct copy of
; %inner.end for the path by which it is reached.
; CHECK-LABEL: define void @test18(
; CHECK:     left:
; CHECK:       %l = cleanuppad []
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     right:
; CHECK:       catchpad []
; CHECK:           to label %[[RIGHT_CATCH:.+]] unwind label %[[RIGHT_END:.+]]
; CHECK:     [[RIGHT_CATCH]]:
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[RIGHT_END]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       [[I_RIGHT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_CATCH_RIGHT:.+]] unwind label %[[INNER_SIBLING_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       [[I_LEFT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_CATCH_LEFT:.+]] unwind label %[[INNER_SIBLING_LEFT:.+]]
; CHECK:     [[INNER_CATCH_RIGHT]]:
; CHECK:       invoke void @f()
; CHECK:         to label %[[UNREACHABLE_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_CATCH_LEFT]]:
; CHECK:       invoke void @f()
; CHECK:         to label %[[UNREACHABLE_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_SIBLING_RIGHT]]:
; CHECK:       [[IS_RIGHT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_SIBLING_CATCH_RIGHT:.+]] unwind label %[[INNER_END_RIGHT]]
; CHECK:     [[INNER_SIBLING_LEFT]]:
; CHECK:       [[IS_LEFT:\%.+]] = catchpad []
; CHECK:           to label %[[INNER_SIBLING_CATCH_LEFT:.+]] unwind label %[[INNER_END_LEFT]]
; CHECK:     [[INNER_SIBLING_CATCH_RIGHT]]:
; CHECK:       call void @h(i32 0)
; CHECK:       unreachable
; CHECK:     [[INNER_SIBLING_CATCH_LEFT]]:
; CHECK:       call void @h(i32 0)
; CHECK:       unreachable
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       catchendpad unwind label %[[RIGHT_END]]


define void @test19() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %invoke.cont unwind label %left
invoke.cont:
  invoke void @f()
    to label %exit unwind label %right
left:
  %l = cleanuppad []
  br label %shared
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind to caller
shared:
  invoke void @f()
    to label %unreachable unwind label %inner
unreachable:
  unreachable
inner:
  %i = cleanuppad []
  invoke void @f()
    to label %unreachable unwind label %inner.end
inner.end:
  cleanupendpad %i unwind label %right.end
exit:
  ret void
}
; This case tests the scenario where an invoke in a funclet with multiple
; parents unwinds to a cleanup end pad for the funclet.  The unwind destination
; for the invoke should map to the correct copy of the cleanup end pad block.
; CHECK-LABEL: define void @test19(
; CHECK:     left:
; CHECK:       %l = cleanuppad []
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     right:
; CHECK:       catchpad []
; CHECK:           to label %[[RIGHT_CATCH:.+]] unwind label %[[RIGHT_END:.+]]
; CHECK:     [[RIGHT_CATCH]]:
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[RIGHT_END]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       [[I_RIGHT:\%.+]] = cleanuppad []
; CHECK:       invoke void @f()
; CHECK:         to label %[[UNREACHABLE_RIGHT:.+]] unwind label %[[INNER_END_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       [[I_LEFT:\%.+]] = cleanuppad []
; CHECK:       invoke void @f()
; CHECK:         to label %[[UNREACHABLE_LEFT:.+]] unwind label %[[INNER_END_LEFT:.+]]
; CHECK:     [[INNER_END_RIGHT]]:
; CHECK:       cleanupendpad [[I_RIGHT]] unwind label %[[RIGHT_END]]
; CHECK:     [[INNER_END_LEFT]]:
; CHECK:       cleanupendpad [[I_LEFT]] unwind to caller

define void @test20() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %invoke.cont unwind label %left
invoke.cont:
  invoke void @f()
    to label %exit unwind label %right
left:
  %l = cleanuppad []
  br label %shared
right:
  catchpad []
    to label %right.catch unwind label %right.end
right.catch:
  br label %shared
right.end:
  catchendpad unwind to caller
shared:
  invoke void @f()
    to label %unreachable unwind label %inner
unreachable:
  unreachable
inner:
  %i = cleanuppad []
  invoke void @f()
    to label %unreachable unwind label %inner.cleanup
inner.cleanup:
  cleanuppad []
  call void @f()
  unreachable
exit:
  ret void
}
; This tests the case where a funclet with multiple parents contains an invoke
; instruction that unwinds to a child funclet.  Here %left and %right are both
; parents of %inner.  Initially %inner is the only parent of %inner.cleanup but
; after %inner is cloned, %inner.cleanup has multiple parents and so it must
; also be cloned.
; CHECK-LABEL: define void @test20(
; CHECK:     left:
; CHECK:       %l = cleanuppad []
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_LEFT:.+]] unwind label %[[INNER_LEFT:.+]]
; CHECK:     right:
; CHECK:       catchpad []
; CHECK:           to label %[[RIGHT_CATCH:.+]] unwind label %[[RIGHT_END:.+]]
; CHECK:     [[RIGHT_CATCH]]:
; CHECK:       invoke void @f()
; CHECK:           to label %[[SHARED_CONT_RIGHT:.+]] unwind label %[[INNER_RIGHT:.+]]
; CHECK:     [[RIGHT_END]]:
; CHECK:       catchendpad unwind to caller
; CHECK:     [[SHARED_CONT_RIGHT]]:
; CHECK:       unreachable
; CHECK:     [[SHARED_CONT_LEFT]]:
; CHECK:       unreachable
; CHECK:     [[INNER_RIGHT]]:
; CHECK:       [[I_RIGHT:\%.+]] = cleanuppad []
; CHECK:       invoke void @f()
; CHECK:         to label %[[UNREACHABLE_RIGHT:.+]] unwind label %[[INNER_CLEANUP_RIGHT:.+]]
; CHECK:     [[INNER_LEFT]]:
; CHECK:       [[I_LEFT:\%.+]] = cleanuppad []
; CHECK:       invoke void @f()
; CHECK:         to label %[[UNREACHABLE_LEFT:.+]] unwind label %[[INNER_CLEANUP_LEFT:.+]]
; CHECK:     [[INNER_CLEANUP_RIGHT]]:
; CHECK:       cleanuppad []
; CHECK:       call void @f()
; CHECK:       unreachable
; CHECK:     [[INNER_CLEANUP_LEFT]]:
; CHECK:       cleanuppad []
; CHECK:       call void @f()
; CHECK:       unreachable


