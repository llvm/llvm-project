; RUN: opt -mtriple=x86_x64-pc-windows-msvc -S -winehprepare  < %s | FileCheck %s

declare i32 @__CxxFrameHandler3(...)

declare void @f()
declare i32 @g()
declare void @h(i32)
declare i1 @b()


define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  ; %x def colors: {entry} subset of use colors; must spill
  %x = call i32 @g()
  invoke void @f()
    to label %noreturn unwind label %catch
catch:
  catchpad []
    to label %noreturn unwind label %endcatch
noreturn:
  ; %x use colors: {entry, cleanup}
  call void @h(i32 %x)
  unreachable
endcatch:
  catchendpad unwind to caller
}
; Need two copies of the call to @h, one under entry and one under catch.
; Currently we generate a load for each, though we shouldn't need one
; for the use in entry's copy.
; CHECK-LABEL: define void @test1(
; CHECK: entry:
; CHECK:   store i32 %x, i32* [[Slot:%[^ ]+]]
; CHECK:   invoke void @f()
; CHECK:     to label %[[EntryCopy:[^ ]+]] unwind label %catch
; CHECK: catch:
; CHECK:   catchpad []
; CHECK-NEXT: to label %[[CatchCopy:[^ ]+]] unwind
; CHECK: [[CatchCopy]]:
; CHECK:   [[LoadX2:%[^ ]+]] = load i32, i32* [[Slot]]
; CHECK:   call void @h(i32 [[LoadX2]]
; CHECK: [[EntryCopy]]:
; CHECK:   [[LoadX1:%[^ ]+]] = load i32, i32* [[Slot]]
; CHECK:   call void @h(i32 [[LoadX1]]


define void @test2() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %cleanup
cleanup:
  cleanuppad []
  br label %exit
exit:
  call void @f()
  ret void
}
; Need two copies of %exit's call to @f -- the subsequent ret is only
; valid when coming from %entry, but on the path from %cleanup, this
; might be a valid call to @f which might dynamically not return.
; CHECK-LABEL: define void @test2(
; CHECK: entry:
; CHECK:   invoke void @f()
; CHECK:     to label %[[exit:[^ ]+]] unwind label %cleanup
; CHECK: cleanup:
; CHECK:   cleanuppad []
; CHECK:   call void @f()
; CHECK-NEXT: unreachable
; CHECK: [[exit]]:
; CHECK:   call void @f()
; CHECK-NEXT: ret void


define void @test3() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %invoke.cont unwind label %catch
invoke.cont:
  invoke void @f()
    to label %exit unwind label %cleanup
catch:
  catchpad [] to label %shared unwind label %endcatch
endcatch:
  catchendpad unwind to caller
cleanup:
  cleanuppad []
  br label %shared
shared:
  call void @f()
  br label %exit
exit:
  ret void
}
; Need two copies of %shared's call to @f (similar to @test2 but
; the two regions here are siblings, not parent-child).
; CHECK-LABEL: define void @test3(
; CHECK:   invoke void @f()
; CHECK:   invoke void @f()
; CHECK:     to label %[[exit:[^ ]+]] unwind
; CHECK: catch:
; CHECK:   catchpad []
; CHECK-NEXT: to label %[[shared:[^ ]+]] unwind
; CHECK: cleanup:
; CHECK:   cleanuppad []
; CHECK:   call void @f()
; CHECK-NEXT: unreachable
; CHECK: [[shared]]:
; CHECK:   call void @f()
; CHECK-NEXT: unreachable
; CHECK: [[exit]]:
; CHECK:   ret void


define void @test4() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %shared unwind label %catch
catch:
  catchpad []
    to label %shared unwind label %endcatch
endcatch:
  catchendpad unwind to caller
shared:
  %x = call i32 @g()
  %i = call i32 @g()
  %zero.trip = icmp eq i32 %i, 0
  br i1 %zero.trip, label %exit, label %loop
loop:
  %i.loop = phi i32 [ %i, %shared ], [ %i.dec, %loop.tail ]
  %b = call i1 @b()
  br i1 %b, label %left, label %right
left:
  %y = call i32 @g()
  br label %loop.tail
right:
  call void @h(i32 %x)
  br label %loop.tail
loop.tail:
  %i.dec = sub i32 %i.loop, 1
  %done = icmp eq i32 %i.dec, 0
  br i1 %done, label %exit, label %loop
exit:
  call void @h(i32 %x)
  unreachable
}
; Make sure we can clone regions that have internal control
; flow and SSA values.  Here we need two copies of everything
; from %shared to %exit.
; CHECK-LABEL: define void @test4(
; CHECK:  entry:
; CHECK:    to label %[[shared_E:[^ ]+]] unwind label %catch
; CHECK:  catch:
; CHECK:    to label %[[shared_C:[^ ]+]] unwind label %endcatch
; CHECK:  [[shared_C]]:
; CHECK:    [[x_C:%[^ ]+]] = call i32 @g()
; CHECK:    [[i_C:%[^ ]+]] = call i32 @g()
; CHECK:    [[zt_C:%[^ ]+]] = icmp eq i32 [[i_C]], 0
; CHECK:    br i1 [[zt_C]], label %[[exit_C:[^ ]+]], label %[[loop_C:[^ ]+]]
; CHECK:  [[shared_E]]:
; CHECK:    [[x_E:%[^ ]+]] = call i32 @g()
; CHECK:    [[i_E:%[^ ]+]] = call i32 @g()
; CHECK:    [[zt_E:%[^ ]+]] = icmp eq i32 [[i_E]], 0
; CHECK:    br i1 [[zt_E]], label %[[exit_E:[^ ]+]], label %[[loop_E:[^ ]+]]
; CHECK:  [[loop_C]]:
; CHECK:    [[iloop_C:%[^ ]+]] = phi i32 [ [[i_C]], %[[shared_C]] ], [ [[idec_C:%[^ ]+]], %[[looptail_C:[^ ]+]] ]
; CHECK:    [[b_C:%[^ ]+]] = call i1 @b()
; CHECK:    br i1 [[b_C]], label %[[left_C:[^ ]+]], label %[[right_C:[^ ]+]]
; CHECK:  [[loop_E]]:
; CHECK:    [[iloop_E:%[^ ]+]] = phi i32 [ [[i_E]], %[[shared_E]] ], [ [[idec_E:%[^ ]+]], %[[looptail_E:[^ ]+]] ]
; CHECK:    [[b_E:%[^ ]+]] = call i1 @b()
; CHECK:    br i1 [[b_E]], label %[[left_E:[^ ]+]], label %[[right_E:[^ ]+]]
; CHECK:  [[left_C]]:
; CHECK:    [[y_C:%[^ ]+]] = call i32 @g()
; CHECK:    br label %[[looptail_C]]
; CHECK:  [[left_E]]:
; CHECK:    [[y_E:%[^ ]+]] = call i32 @g()
; CHECK:    br label %[[looptail_E]]
; CHECK:  [[right_C]]:
; CHECK:    call void @h(i32 [[x_C]])
; CHECK:    br label %[[looptail_C]]
; CHECK:  [[right_E]]:
; CHECK:    call void @h(i32 [[x_E]])
; CHECK:    br label %[[looptail_E]]
; CHECK:  [[looptail_C]]:
; CHECK:    [[idec_C]] = sub i32 [[iloop_C]], 1
; CHECK:    [[done_C:%[^ ]+]] = icmp eq i32 [[idec_C]], 0
; CHECK:    br i1 [[done_C]], label %[[exit_C]], label %[[loop_C]]
; CHECK:  [[looptail_E]]:
; CHECK:    [[idec_E]] = sub i32 [[iloop_E]], 1
; CHECK:    [[done_E:%[^ ]+]] = icmp eq i32 [[idec_E]], 0
; CHECK:    br i1 [[done_E]], label %[[exit_E]], label %[[loop_E]]
; CHECK:  [[exit_C]]:
; CHECK:    call void @h(i32 [[x_C]])
; CHECK:    unreachable
; CHECK:  [[exit_E]]:
; CHECK:    call void @h(i32 [[x_E]])
; CHECK:    unreachable


define void @test5() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %outer
outer:
  %o = cleanuppad []
  %x = call i32 @g()
  invoke void @f()
    to label %outer.ret unwind label %inner
inner:
  %i = catchpad []
    to label %inner.catch unwind label %inner.endcatch
inner.catch:
  catchret %i to label %outer.post-inner
inner.endcatch:
  catchendpad unwind to caller
outer.post-inner:
  call void @h(i32 %x)
  br label %outer.ret
outer.ret:
  cleanupret %o unwind to caller
exit:
  ret void
}
; Simple nested case (catch-inside-cleanup).  Nothing needs
; to be cloned.  The def and use of %x are both in %outer
; and so don't need to be spilled.
; CHECK-LABEL: define void @test5(
; CHECK:      outer:
; CHECK:        %x = call i32 @g()
; CHECK-NEXT:   invoke void @f()
; CHECK-NEXT:     to label %outer.ret unwind label %inner
; CHECK:      inner:
; CHECK:          to label %inner.catch unwind label %inner.endcatch
; CHECK:      inner.catch:
; CHECK-NEXT:   catchret %i to label %outer.post-inner
; CHECK:      outer.post-inner:
; CHECK-NEXT:   call void @h(i32 %x)
; CHECK-NEXT:   br label %outer.ret


define void @test6() personality i32 (...)* @__CxxFrameHandler3 {
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
; CHECK-LABEL: define void @test6(
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


define void @test7() personality i32 (...)* @__CxxFrameHandler3 {
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
; Another case of a two-parent child (like @test6), this time
; with the join at the entry itself instead of following a
; non-pad join.
; CHECK-LABEL: define void @test7(
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


define void @test8() personality i32 (...)* @__CxxFrameHandler3 {
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
; CHECK-LABEL: define void @test8(
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


define void @test9() personality i32 (...)* @__CxxFrameHandler3 {
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
; CHECK-LABEL: define void @test9(
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


define void @test10() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %unreachable unwind label %inner
inner:
  %cleanup = cleanuppad []
  ; make sure we don't overlook this cleanupret and try to process
  ; successor %outer as a child of inner.
  cleanupret %cleanup unwind label %outer
outer:
  %catch = catchpad [] to label %catch.body unwind label %endpad
catch.body:
  catchret %catch to label %exit
endpad:
  catchendpad unwind to caller
exit:
  ret void
unreachable:
  unreachable
}
; CHECK-LABEL: define void @test10(
; CHECK-NEXT: entry:
; CHECK-NEXT:   invoke
; CHECK-NEXT:     to label %unreachable unwind label %inner
; CHECK:      inner:
; CHECK-NEXT:   %cleanup = cleanuppad
; CHECK-NEXT:   cleanupret %cleanup unwind label %outer
; CHECK:      outer:
; CHECK-NEXT:   %catch = catchpad []
; CHECK-NEXT:	      to label %catch.body unwind label %endpad
; CHECK:      catch.body:
; CHECK-NEXT:   catchret %catch to label %exit
; CHECK:      endpad:
; CHECK-NEXT:   catchendpad unwind to caller
; CHECK:      exit:
; CHECK-NEXT:   ret void

define void @test11() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %cleanup.outer
cleanup.outer:
  %outer = cleanuppad []
  invoke void @f()
    to label %outer.cont unwind label %cleanup.inner
outer.cont:
  br label %merge
cleanup.inner:
  %inner = cleanuppad []
  br label %merge
merge:
  invoke void @f()
    to label %unreachable unwind label %merge.end
unreachable:
  unreachable
merge.end:
  cleanupendpad %outer unwind to caller
exit:
  ret void
}
; merge.end will get cloned for outer and inner, but is implausible
; from inner, so the invoke @f() in inner's copy of merge should be
; rewritten to call @f()
; CHECK-LABEL: define void @test11()
; CHECK:      %inner = cleanuppad []
; CHECK-NEXT: call void @f()
; CHECK-NEXT: unreachable

define void @test12() personality i32 (...)* @__CxxFrameHandler3 !dbg !5 {
entry:
  invoke void @f()
    to label %cont unwind label %left, !dbg !8
cont:
  invoke void @f()
    to label %exit unwind label %right
left:
  cleanuppad []
  br label %join
right:
  cleanuppad []
  br label %join
join:
  ; This call will get cloned; make sure we can handle cloning
  ; instructions with debug metadata attached.
  call void @f(), !dbg !9
  unreachable
exit:
  ret void
}

; CHECK-LABEL: define void @test13()
; CHECK: ret void
define void @test13() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  ret void

unreachable:
  cleanuppad []
  unreachable
}

define void @test14() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
    to label %exit unwind label %catch1.pad
catch1.pad:
  %catch1 = catchpad [i32 1]
    to label %catch1.body unwind label %catch2.pad
catch1.body:
  invoke void @h(i32 1)
    to label %catch1.body2 unwind label %catch.end
catch1.body2:
  invoke void @f()
    to label %catch1.ret unwind label %cleanup1.pad
cleanup1.pad:
  %cleanup1 = cleanuppad []
  call void @f()
  cleanupret %cleanup1 unwind label %catch.end
catch1.ret:
  catchret %catch1 to label %exit
catch2.pad:
  %catch2 = catchpad [i32 2]
    to label %catch2.body unwind label %catch.end
catch2.body:
  invoke void @h(i32 2)
    to label %catch2.body2 unwind label %catch.end
catch2.body2:
  invoke void @f()
    to label %catch2.ret unwind label %cleanup2.pad
cleanup2.pad:
  %cleanup2 = cleanuppad []
  call void @f()
  cleanupret %cleanup2 unwind label %catch.end
catch2.ret:
  catchret %catch2 to label %exit
catch.end:
  catchendpad unwind to caller
exit:
  ret void
}
; Make sure we don't clone the catchendpad even though the
; cleanupendpads targeting it would naively imply that it
; should get their respective parent colors (catch1 and catch2),
; as well as its properly getting the root function color.  The
; references from the invokes ensure that if we did make clones
; for each catch, they'd be reachable, as those invokes would get
; rewritten
; CHECK-LABEL: define void @test14()
; CHECK-NOT:  catchendpad
; CHECK:      invoke void @h(i32 1)
; CHECK-NEXT:   unwind label %catch.end
; CHECK-NOT:  catchendpad
; CHECK:      invoke void @h(i32 2)
; CHECK-NEXT:   unwind label %catch.end
; CHECK-NOT:   catchendpad
; CHECK:     catch.end:
; CHECK-NEXT:  catchendpad
; CHECK-NOT:   catchendpad

;; Debug info (from test12)

; Make sure the DISubprogram doesn't get cloned
; CHECK-LABEL: !llvm.module.flags
; CHECK-NOT: !DISubprogram
; CHECK: !{{[0-9]+}} = distinct !DISubprogram(name: "test12"
; CHECK-NOT: !DISubprogram
!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "compiler", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !3, subprograms: !4)
!2 = !DIFile(filename: "test.cpp", directory: ".")
!3 = !{}
!4 = !{!5}
!5 = distinct !DISubprogram(name: "test12", scope: !2, file: !2, type: !6, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, variables: !3)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 1, scope: !5)
!9 = !DILocation(line: 2, scope: !5)
